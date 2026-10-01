import copy

import numpy as np
import pytest
import torch

from relative_dp.config import TRAIN_CONFIG
from relative_dp.dataset import Normalizer, WindowDataset
from relative_dp.model import initialized_policy, make_ema, update_ema, masked_epsilon_loss, state_hash, load_policy
from relative_dp.train import Trainer, atomic_torch_save, _recover_log, _restore_milestone_checkpoint


def tiny_config():
    return dict(TRAIN_CONFIG, unet_down_dims=[8,16,32], diffusion_step_embed_dim=16,
                batch_size=4, microbatch_size=2, torch_compile=False)


def tiny_dataset(representation='raw'):
    rng = np.random.default_rng(55)
    episodes = [dict(obs=rng.normal(size=(10,39)).astype(np.float32),
                     actions=rng.uniform(-1,1,size=(9,4)).astype(np.float32))]
    norm = Normalizer.fit(episodes, representation, ['train20-a'], {'train20-a':'sha'})
    return WindowDataset(episodes, norm)


def test_scheduler_shape_gradient_mask_and_ema():
    torch.set_num_threads(1)
    config = tiny_config()
    policy = initialized_policy(39, config, 'cpu')
    torch.testing.assert_close(policy.train_scheduler.betas, policy.inference_scheduler.betas, rtol=0, atol=0)
    obs = torch.zeros(2,2,39)
    actions = torch.zeros(2,16,4)
    noisy, steps, noise = policy.noise_targets(actions, torch.Generator().manual_seed(4))
    output = policy(noisy, steps, obs)
    assert output.shape == actions.shape
    output.retain_grad()
    mask = torch.zeros(2,16)
    mask[:, :2] = 1
    loss = masked_epsilon_loss(output, noise, mask)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in policy.parameters())
    assert output.grad[:,2:].count_nonzero() == 0
    ema = make_ema(policy)
    before = next(ema.parameters()).clone()
    with torch.no_grad():
        next(policy.parameters()).add_(2)
    update_ema(ema, policy, .995)
    torch.testing.assert_close(next(ema.parameters()), before + .01, rtol=1e-5, atol=1e-7)
    sampled = policy.predict_action(obs, torch.Generator().manual_seed(8))
    repeated = policy.predict_action(obs, torch.Generator().manual_seed(8))
    assert sampled.shape == (2,16,4) and sampled.abs().max() <= 1
    assert len(policy.inference_scheduler.timesteps) == 16
    torch.testing.assert_close(sampled, repeated, rtol=0, atol=0)


def test_resume_reproduces_uninterrupted_optimizer_rng_and_ema(tmp_path):
    config = tiny_config()
    full = Trainer(tiny_dataset(), config, 'cpu')
    full.update()
    full.update()
    checkpoint_path = tmp_path / 'resume.pt'
    metadata = dict(run_id='debug', manifest_hash='frozen', code_hash='source')
    atomic_torch_save(checkpoint_path, full.checkpoint(metadata))
    expected = [full.update(), full.update()]
    resumed = Trainer(tiny_dataset(), config, 'cpu')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    resumed.restore(checkpoint, metadata)
    actual = [resumed.update(), resumed.update()]
    assert [r['loss'] for r in actual] == [r['loss'] for r in expected]
    assert resumed.pairing_digest == full.pairing_digest
    assert state_hash(resumed.policy.state_dict()) == state_hash(full.policy.state_dict())
    assert state_hash(resumed.ema.state_dict()) == state_hash(full.ema.state_dict())
    torch.testing.assert_close(resumed.loader_rng.get_state(), full.loader_rng.get_state(), rtol=0, atol=0)
    torch.testing.assert_close(resumed.diffusion_rng.get_state(), full.diffusion_rng.get_state(), rtol=0, atol=0)
    with pytest.raises(ValueError, match='identity mismatch'):
        resumed.restore(checkpoint, dict(metadata, manifest_hash='changed'))


def test_raw_relative_share_initial_weights_and_sampling_streams():
    raw = Trainer(tiny_dataset('raw'), tiny_config(), 'cpu')
    relative = Trainer(tiny_dataset('relative'), tiny_config(), 'cpu')
    assert raw.initial_weights_hash == relative.initial_weights_hash
    assert sum(p.numel() for p in raw.policy.parameters()) == sum(p.numel() for p in relative.policy.parameters())
    raw.update()
    relative.update()
    assert raw.pairing_digest == relative.pairing_digest
    assert state_hash(raw.policy.state_dict()) != state_hash(relative.policy.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA compile is an optional platform capability')
def test_compiled_cuda_resume_and_ema_inference(tmp_path):
    config = dict(TRAIN_CONFIG, torch_compile=True, train_compile_mode='reduce-overhead',
                  inference_torch_compile=True, inference_compile_mode='reduce-overhead')
    full = Trainer(tiny_dataset(), config, 'cuda')
    full.update()
    full.update()
    path = tmp_path / 'compiled_resume.pt'
    metadata = dict(run_id='debug_cuda', manifest_hash='frozen', code_hash='source')
    atomic_torch_save(path, full.checkpoint(metadata))
    expected = full.update()
    expected_model = state_hash(full.policy.state_dict())
    expected_ema = state_hash(full.ema.state_dict())
    resumed = Trainer(tiny_dataset(), config, 'cuda')
    resumed.restore(torch.load(path, map_location='cuda', weights_only=False), metadata)
    actual = resumed.update()
    assert expected['loss'] == actual['loss']
    assert expected['pairing_digest'] == actual['pairing_digest']
    assert expected_model == state_hash(resumed.policy.state_dict())
    assert expected_ema == state_hash(resumed.ema.state_dict())
    loaded = load_policy(path, 'cuda')
    assert state_hash(loaded.policy.state_dict()) == state_hash(torch.load(path, weights_only=False)['ema'])
    observations = np.zeros((2,39), np.float32)
    first = loaded.predict_actions(observations, torch.Generator(device='cuda').manual_seed(67))
    second = loaded.predict_actions(observations, torch.Generator(device='cuda').manual_seed(67))
    assert first.shape == (16,4) and np.isfinite(first).all() and np.abs(first).max() <= 1
    np.testing.assert_array_equal(first, second)


def test_interrupted_milestone_copy_and_partial_log_are_recoverable(tmp_path):
    trainer = Trainer(tiny_dataset(), tiny_config(), 'cpu')
    trainer.update()
    metadata = dict(run_id='debug', manifest_hash='frozen', code_hash='source')
    latest = tmp_path / 'latest.pt'
    atomic_torch_save(latest, trainer.checkpoint(metadata))
    # Simulate death immediately after the atomic latest rename, before milestone save.
    checkpoint = torch.load(latest, weights_only=False)
    resumed = Trainer(tiny_dataset(), tiny_config(), 'cpu')
    resumed.restore(checkpoint, metadata)
    _restore_milestone_checkpoint(tmp_path, checkpoint, [1])
    milestone = torch.load(tmp_path / 'checkpoints/step_000001.pt', weights_only=False)
    assert milestone['step'] == 1
    assert state_hash(milestone['ema']) == state_hash(checkpoint['ema'])
    assert milestone['pairing_digest'] == checkpoint['pairing_digest']
    log = tmp_path / 'train.jsonl'
    log.write_text('{"step": 1, "loss": 1}\n{"step": 2, "loss": 0.5}\n{"step": 3, "loss":')
    _recover_log(log, step=1)
    assert log.read_text() == '{"step": 1, "loss": 1}\n'
    archives = list(tmp_path.glob('train_uncommitted_*.jsonl'))
    assert len(archives) == 1 and archives[0].read_text().endswith('{"step": 3, "loss":')
    log.write_text('{invalid}\n{"step": 1}\n')
    with pytest.raises(ValueError):
        _recover_log(log, step=1)
