import numpy as np
import torch

from relative_dp.dataset import Normalizer, WindowDataset
from relative_dp.representation import inverse_observation, transform_observation


def indexed_episode(length, offset=0):
    obs = np.zeros((length+1, 39), np.float32)
    obs[:, 0] = np.arange(length+1) + offset
    obs[:, 4:7] = [.1, .2, .3]
    obs[:, 18] = np.maximum(np.arange(length+1)-1, 0) + offset
    obs[:, 22:25] = [.1, .2, .3]
    actions = np.repeat((np.arange(length) / 100 + offset / 1000)[:, None], 4, axis=1).astype(np.float32)
    return dict(obs=obs, actions=actions)


def test_alignment_start_tail_and_episode_boundary():
    episodes = [indexed_episode(4), indexed_episode(3, 20)]
    norm = Normalizer(np.zeros(39, np.float32), np.ones(39, np.float32), ['a','b'], {}, [], 'raw', 7)
    dataset = WindowDataset(episodes, norm, horizon=16)
    assert dataset.index == [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)]
    torch.testing.assert_close(dataset.observations[0,:,0], torch.tensor([0.,0.]))
    torch.testing.assert_close(dataset.observations[2,:,0], torch.tensor([1.,2.]))
    torch.testing.assert_close(dataset.observations[4,:,0], torch.tensor([20.,20.]))
    np.testing.assert_allclose(dataset.actions[2,:2], episodes[0]['actions'][2:])
    assert dataset.valid_mask[2].sum().item() == 2
    assert dataset.actions[2,2:].count_nonzero().item() == 0
    assert dataset.valid_mask[3].sum().item() == 1
    assert dataset.actions[3,1:].count_nonzero().item() == 0


def test_normalization_sources_ood_and_representation_causality():
    episode = indexed_episode(4)
    original_actions = episode['actions'].copy()
    normalizer = Normalizer.fit([episode], 'relative', ['selected'], {'selected':'source-sha'})
    assert normalizer.source_ids == ['selected'] and normalizer.source_hashes == {'selected':'source-sha'}
    assert normalizer.count == 4
    assert normalizer.std.min() >= .001
    # Terminal observation is not an eligible conditioning state.
    altered = dict(obs=episode['obs'].copy(), actions=episode['actions'].copy())
    altered['obs'][-1] = 1e5
    same = Normalizer.fit([altered], 'relative', ['selected'], {'selected':'source-sha'})
    np.testing.assert_array_equal(normalizer.mean, same.mean)
    extreme = episode['obs'][0].copy()
    extreme[0] = 100
    assert normalizer.normalize(extreme)[0] > 50
    transformed = transform_observation(episode['obs'], 'relative')
    np.testing.assert_allclose(inverse_observation(transformed), episode['obs'], atol=1e-6)
    changed_future = episode['obs'].copy()
    changed_future[2:] = 500
    np.testing.assert_array_equal(transform_observation(changed_future, 'relative')[:2], transformed[:2])
    WindowDataset([episode], normalizer)
    np.testing.assert_array_equal(original_actions, episode['actions'])
