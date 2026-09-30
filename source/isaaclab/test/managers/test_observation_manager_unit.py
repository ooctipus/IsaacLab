# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for observation managers."""

from __future__ import annotations

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none
import inspect
from typing import TYPE_CHECKING, cast

import pytest
import torch

from isaaclab.managers import (
    ManagerTermBase,
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
    SceneEntityCfg,
)
from isaaclab.utils import modifiers
from isaaclab.utils.configclass import configclass

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def dummy_observation(env: DummyEnv) -> torch.Tensor:
    """Return the dummy environment observation."""
    env.term_calls += 1
    return env.observation


def selected_body_observation(env: DummyEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return a body-sized observation without participating in shape discovery."""
    env.term_calls += 1
    return env.observation.expand(-1, 6 * len(sensor_cfg.body_ids))


def _selected_body_output_shape(env: DummyEnv, sensor_cfg: SceneEntityCfg, **_: object) -> tuple[int, ...]:
    env.shape_calls += 1
    return (6 * len(sensor_cfg.body_ids),)


selected_body_observation._output_shape = _selected_body_output_shape


class DummySimulation:
    """Minimal playing simulation double."""

    def is_playing(self) -> bool:
        """Return whether the simulated timeline is playing."""
        return True


class DummyEnv:
    """Minimal environment double used by :class:`ObservationManager`."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.sim = DummySimulation()
        self.observation = torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1)
        self.term_calls = 0
        self.term_resets = 0
        self.shape_calls = 0
        self.scene = {
            "sensor": type(
                "DummyEntity",
                (),
                {
                    "body_names": ["left", "middle", "right"],
                    "num_bodies": 3,
                    "find_bodies": lambda self, names, preserve_order=False: ([0, 2], names),
                },
            )()
        }


class DeclaredShapeObservation(ManagerTermBase):
    """Observation term whose shape is known without evaluating it."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._output_shape = (3,)

    def reset(self, env_ids=None) -> None:
        self._env.term_resets += 1

    def __call__(self, env: DummyEnv) -> torch.Tensor:
        env.term_calls += 1
        return env.observation.expand(-1, 3)


class StatefulBiasModifier(modifiers.ModifierBase):
    """Stateful modifier used to verify lazy callable resolution."""

    def __init__(self, cfg: modifiers.ModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        super().__init__(cfg, data_dim, device)
        self.value = cfg.params["value"]
        self.reset_count = 0

    def reset(self, env_ids=None) -> None:
        self.reset_count += 1

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data + self.value


class InvalidModifier:
    """Class with the modifier constructor contract but the wrong base type."""

    def __init__(self, cfg, data_dim, device):
        pass


@configclass
class HistoryObservationsCfg:
    """Observation configuration with group-level history."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group configuration."""

        dummy: ObservationTermCfg = ObservationTermCfg(func=dummy_observation)

        def __post_init__(self):
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


def test_class_modifier_roundtrip_preserves_func_and_params():
    """Reproduce #6067 with a class modifier and non-empty parameters."""
    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy.modifiers = [modifiers.ModifierCfg(func=StatefulBiasModifier, params={"value": 2.0})]
    cfg.from_dict(cfg.to_dict())
    term_cfg = cfg.policy.dummy
    assert term_cfg.modifiers is not None
    modifier_cfg = term_cfg.modifiers[0]
    assert isinstance(modifier_cfg, modifiers.ModifierCfg)
    assert isinstance(modifier_cfg.func, str)
    assert modifier_cfg.params == {"value": 2.0}

    env = DummyEnv()
    manager = ObservationManager(cfg, cast("ManagerBasedEnv", env))
    prepared_term_cfg = manager.cfg.policy.dummy
    assert prepared_term_cfg.modifiers is not None
    prepared_modifier_cfg = prepared_term_cfg.modifiers[0]
    assert isinstance(prepared_modifier_cfg.func, StatefulBiasModifier)
    observations = manager.compute()["policy"]
    torch.testing.assert_close(observations, env.observation + 2.0)

    manager.reset()
    assert prepared_modifier_cfg.func.reset_count == 1


def test_stateless_modifier_cfg_roundtrip_preserves_signature_validation():
    """A stateless modifier remains callable and inspectable after a configuration round-trip."""
    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy.modifiers = [modifiers.ModifierCfg(func=modifiers.bias, params={"value": 2.0})]
    cfg.from_dict(cfg.to_dict())

    env = DummyEnv()
    manager = ObservationManager(cfg, cast("ManagerBasedEnv", env))
    observations = manager.compute()["policy"]
    torch.testing.assert_close(observations, env.observation + 2.0)


def test_class_modifier_validates_constructed_instance():
    """Class modifier validation checks the constructed object."""
    cfg = HistoryObservationsCfg()
    cfg.policy.history_length = None
    cfg.policy.dummy.modifiers = [modifiers.ModifierCfg(func=InvalidModifier)]
    cfg.from_dict(cfg.to_dict())

    with pytest.raises(TypeError, match="is not an instance of 'ModifierBase'"):
        ObservationManager(cfg, cast("ManagerBasedEnv", DummyEnv()))


def test_modifier_resolution_stays_out_of_observation_manager():
    """Observation-specific code receives resolved modifier callables from ``ManagerBase``."""
    source = inspect.getsource(ObservationManager._prepare_terms)
    assert "inspect.isclass(mod_cfg.func)" in source
    assert "string_to_callable" not in source


def test_modifier_base_cfg_marker_does_not_exist():
    """Stateful modifiers must not require a marker configuration subtype."""
    assert not hasattr(modifiers, "ModifierBaseCfg")


@configclass
class DeclaredShapeObservationsCfg:
    """Observation configuration with a class-declared output shape."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        declared: ObservationTermCfg = ObservationTermCfg(func=DeclaredShapeObservation)

    policy: PolicyCfg = PolicyCfg()


@configclass
class ResolvedShapeObservationsCfg:
    """Observation configuration whose shape depends on a resolved scene entity."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        selected: ObservationTermCfg = ObservationTermCfg(
            func=selected_body_observation,
            params={"sensor_cfg": SceneEntityCfg("sensor", body_names=["left", "right"])},
        )

    policy: PolicyCfg = PolicyCfg()


def test_declared_shape_skips_construction_probe_and_reset():
    """A declared term shape must not execute or reset the term during manager construction."""
    env = DummyEnv()
    manager = ObservationManager(DeclaredShapeObservationsCfg(), cast("ManagerBasedEnv", env))

    assert env.term_calls == 0
    assert env.term_resets == 0
    assert manager.group_obs_term_dim["policy"] == [(3,)]

    observations = manager.compute()["policy"]
    assert env.term_calls == 1
    assert observations.shape == (env.num_envs, 3)


def test_resolved_shape_invokes_only_resolver_during_construction():
    """Dynamic shapes see resolved entity ids without evaluating the observation."""
    env = DummyEnv()
    manager = ObservationManager(ResolvedShapeObservationsCfg(), cast("ManagerBasedEnv", env))

    assert env.shape_calls == 1
    assert env.term_calls == 0
    assert manager.group_obs_term_dim["policy"] == [(12,)]

    observations = manager.compute()["policy"]
    assert env.term_calls == 1
    assert observations.shape == (env.num_envs, 12)


def test_legacy_term_retains_single_construction_probe():
    """Pure legacy functions without shape metadata retain one-call inference."""
    env = DummyEnv()
    ObservationManager(HistoryObservationsCfg(), cast("ManagerBasedEnv", env))

    assert env.term_calls == 1


def test_compute_updates_history_only_when_requested():
    """Observation history changes only when ``update_history`` is enabled."""
    env = DummyEnv()
    manager = ObservationManager(HistoryObservationsCfg(), cast("ManagerBasedEnv", env))
    history = manager._group_obs_term_history_buffer["policy"]["dummy"]

    torch.testing.assert_close(history.current_length, torch.zeros(env.num_envs, dtype=torch.int64))

    manager.compute()
    torch.testing.assert_close(history.current_length, torch.zeros(env.num_envs, dtype=torch.int64))

    manager.compute(update_history=True)
    torch.testing.assert_close(history.current_length, torch.ones(env.num_envs, dtype=torch.int64))
    history_after_update = history.buffer.clone()

    env.observation.add_(10.0)
    observations = manager.compute()
    policy_observation = observations["policy"]
    assert isinstance(policy_observation, torch.Tensor)
    torch.testing.assert_close(history.current_length, torch.ones(env.num_envs, dtype=torch.int64))
    torch.testing.assert_close(history.buffer, history_after_update)
    torch.testing.assert_close(policy_observation, history_after_update.reshape(env.num_envs, -1))

    manager.compute(update_history=True)
    torch.testing.assert_close(history.current_length, torch.full((env.num_envs,), 2, dtype=torch.int64))
    torch.testing.assert_close(history.buffer[:, -1], env.observation)
