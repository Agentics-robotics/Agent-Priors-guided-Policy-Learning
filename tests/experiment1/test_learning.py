"""Public framework checks use synthetic fixtures, never API candidate authorship."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiment1.baselines import RulePriorDesign, VanillaDesign, build_baseline
from experiment1.contracts import CandidateDesign, causal_context
from experiment1.evaluation import run_episode, summarize
from experiment1.learning import (COMMON_TRAIN_CONFIG, LoadedPolicy, RawNormalizer, Trainer,
                                  predict_actions, train, validate_design)
from relative_dp.model import state_hash


def spec(task="stick-push"):
    fields = [dict(name=name, slice=location) for name, location in (
        ("current.hand_xyz", [0, 3]), ("current.object_xyz", [4, 7]),
        ("current.object2_xyz", [11, 14]), ("goal_xyz", [36, 39]))]
    return dict(task=task, observation_schema=dict(raw_dim=39, fields=fields), action_schema=dict(dim=4))


def support():
    rng = np.random.default_rng(94)
    return [dict(obs=rng.normal(size=(6, 39)).astype(np.float32),
                 actions=rng.uniform(-1, 1, (5, 4)).astype(np.float32)) for _ in range(2)]


def make_trainer(design_type=VanillaDesign, updates=4):
    common = spec()
    return Trainer(design_type(common, {}), common, support(), ["e0", "e1"], {"e0": "h0", "e1": "h1"},
                   "cpu", debug_updates=updates)


def test_fixed_recipe_and_support_statistics_exclude_terminal():
    cfg = COMMON_TRAIN_CONFIG
    assert (cfg["train_updates"], cfg["batch_size"], cfg["train_seed"]) == (20000, 128, 0)
    assert (cfg["n_obs_steps"], cfg["prediction_horizon"], cfg["n_action_steps"]) == (2, 16, 4)
    episodes = support()
    expected = np.concatenate([e["obs"][:-1] for e in episodes]).mean(0)
    for episode in episodes:
        episode["obs"][-1] = 1e8
    norm = RawNormalizer.fit(episodes, ["a", "b"], {"a": "a", "b": "b"})
    np.testing.assert_allclose(norm.mean, expected, atol=1e-7)
    with pytest.raises(ValueError, match="exactly"):
        RawNormalizer.fit(episodes, ["a", "b"], {"a": "a", "b": "b", "hidden": "x"})


def test_fixed_baselines_preserve_raw_and_native_action():
    for task in ("stick-push", "drawer"):
        common = spec(task)
        model = RulePriorDesign(common, {})
        trainer = Trainer(model, common, support(), ["a", "b"], {"a": "a", "b": "b"}, debug_updates=0)
        history = trainer.dataset.observations[:2]
        context = causal_context(trainer.dataset.raw_history[:2])
        condition = model.condition(history, context)
        assert torch.equal(condition[..., :39], history)
        assert condition.shape == (2, 2, 71)
        actions = trainer.dataset.actions[:2]
        assert torch.equal(model.encode_actions(actions, context), actions)
        assert torch.equal(model.decode_actions(actions, context), actions)
        assert trainer.identity["parameter_count"] < 3 * trainer.identity["baseline_parameter_count"]
        if task == "drawer":
            assert model.presence.tolist() == [1, 1, 0]
            assert torch.count_nonzero(model.relations(context["raw_history"])[..., 6:]) == 0


def test_interface_check_has_zero_optimizer_updates_and_no_scores():
    common = spec()
    receipt = validate_design(VanillaDesign(common, {}), common, support(), ["a", "b"], {"a": "a", "b": "b"},
                              rebuild_design=lambda: VanillaDesign(common, {}))
    assert receipt["status"] == "valid"
    assert receipt["optimizer_updates"] == 0
    assert receipt["backward_checks"] == 1
    assert receipt["inference_calls"] == 3 and receipt["deployment_state_roundtrip"]
    assert not {"loss", "success_rate", "dev_score"}.intersection(receipt)


def test_exact_checkpoint_restore_matches_uninterrupted_updates():
    full = make_trainer()
    full.update()
    full.update()
    checkpoint = deepcopy(full.checkpoint({"slot": "test"}))
    expected = [full.update(), full.update()]
    resumed = make_trainer()
    resumed.restore(checkpoint, {"slot": "test"})
    actual = [resumed.update(), resumed.update()]
    assert [x["loss"] for x in actual] == [x["loss"] for x in expected]
    assert resumed.pairing_digest == full.pairing_digest
    assert state_hash(resumed.policy.state_dict()) == state_hash(full.policy.state_dict())
    assert state_hash(resumed.ema.state_dict()) == state_hash(full.ema.state_dict())
    with pytest.raises(ValueError, match="identity mismatch"):
        make_trainer().restore(checkpoint, {"slot": "other"})


def test_training_resume_and_complete_artifact_are_checked(tmp_path):
    common = spec()
    args = (common, support(), ["a", "b"], {"a": "a", "b": "b"}, tmp_path, "cpu")
    paused = train(VanillaDesign(common, {}), *args, identity={"slot": "fixture"}, debug_updates=2, stop_after=1)
    assert paused["status"] == "paused"
    complete = train(VanillaDesign(common, {}), *args, identity={"slot": "fixture"}, debug_updates=2)
    assert complete["step"] == 2 and complete["identity"]["debug"]
    reused = train(VanillaDesign(common, {}), *args, identity={"slot": "fixture"}, debug_updates=2)
    assert reused == complete
    with (tmp_path / "latest.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint is invalid"):
        train(VanillaDesign(common, {}), *args, identity={"slot": "fixture"}, debug_updates=2)


def test_loaded_policy_restores_final_ema_and_causal_history():
    trainer = make_trainer(updates=0)
    checkpoint = trainer.checkpoint({"fixture": True})
    loaded = LoadedPolicy(VanillaDesign(spec(), {}), checkpoint)
    history = trainer.dataset.raw_history[0].numpy()
    expected = predict_actions(trainer.ema, trainer.dataset.normalizer, history, torch.Generator().manual_seed(5), "cpu")
    actual = loaded.actions(history, torch.Generator().manual_seed(5))
    np.testing.assert_array_equal(expected, actual)


class BrokenFrame(VanillaDesign):
    def encode_actions(self, actions, context):
        return actions * 2


def test_invalid_candidate_frame_is_interface_diagnostic():
    common = spec()
    with pytest.raises(ValueError, match="roundtrip"):
        validate_design(BrokenFrame(common, {}), common, support(), ["a", "b"], {"a": "a", "b": "b"},
                        rebuild_design=lambda: BrokenFrame(common, {}))


class MissingDeploymentState(VanillaDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.fitted_offset = 0.

    def fit_support(self, episodes, common_spec):
        self.fitted_offset = float(np.mean(episodes[0]["obs"][:-1]))

    def condition(self, history, context):
        return history + self.fitted_offset


def test_fresh_deployment_rejects_unsaved_support_fit_statistics():
    common = spec()
    with pytest.raises(ValueError, match="Fresh deployment state"):
        validate_design(MissingDeploymentState(common, {}), common, support(), ["a", "b"], {"a": "a", "b": "b"},
                        rebuild_design=lambda: MissingDeploymentState(common, {}))


class MutatingCommonConfig(VanillaDesign):
    def condition(self, history, context):
        self.dp.config["train_updates"] = 999999
        return history


def test_candidate_cannot_mutate_public_configuration_from_a_hook():
    trainer = make_trainer(MutatingCommonConfig, updates=1)
    with pytest.raises(ValueError, match="frozen common"):
        trainer.update()
    assert trainer.step == 0


class FixtureEnvironment:
    def reset(self, record):
        self.count = 0
        return np.zeros(39, np.float32), {"success": False}

    def step(self, action):
        assert action.dtype == np.float32 and (np.abs(action) <= 1).all()
        self.count += 1
        return np.full(39, self.count, np.float32), 0., False, False, {"success": self.count == 6}


class FixturePolicy:
    def __init__(self):
        self.histories = []

    def actions(self, history, generator):
        self.histories.append(history.copy())
        return np.full((16, 4), 2, np.float32)


def test_evaluator_keeps_four_step_prefix_and_causal_observations():
    policy = FixturePolicy()
    record = dict(episode_id="fixture", split="C", cell="LH", inference_seed=51)
    result, trace = run_episode(FixtureEnvironment(), policy, record)
    assert result["success"] and result["steps"] == 6 and result["replans"] == 2
    assert result["clipped_steps"] == 6
    assert policy.histories[0][:, 0].tolist() == [0, 0]
    assert policy.histories[1][:, 0].tolist() == [3, 4]
    assert trace["actions"].shape == (6, 4)


def test_equal_cell_weight_is_used_in_dev_score():
    rows = []
    for split in ("IID", "C", "E"):
        for cell, count, success in (("left", 1, True), ("right", 3, False)):
            rows.extend(dict(split=split, cell=cell, success=success, termination_reason="max_steps") for _ in range(count))
    metrics = summarize(rows)
    assert metrics["dev_score"] == .5
    assert metrics["C"]["pooled_success_rate"] == .25
