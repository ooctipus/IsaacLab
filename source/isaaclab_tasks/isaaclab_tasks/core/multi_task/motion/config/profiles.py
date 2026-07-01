# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native timing, reset, observation, and transition-route profiles."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..mdp.commands import MotionStatePayload
    from ..mdp.runtime import MotionRuntime


@configclass
class MotionProfileCfg:
    """One native motion runtime profile with coordinated simulator clocks."""

    @configclass
    class TimingCfg:
        """Physics, control, and timeout clocks."""

        physics_dt: float = MISSING
        """Physics simulation step [s]."""

        control_decimation: int = MISSING
        """Physics steps per applied action."""

        configured_horizon_steps: int = MISSING
        """Configured timeout horizon in applied actions."""

        applied_actions_before_timeout: int = MISSING
        """Observed native number of applied actions through the timeout edge."""

        def __post_init__(self) -> None:
            """Reject inconsistent physical clocks and horizons."""
            if not math.isfinite(self.physics_dt) or self.physics_dt <= 0.0:
                raise ValueError("physics_dt must be finite and positive [s].")
            if self.control_decimation < 1:
                raise ValueError("control_decimation must be positive.")
            if self.configured_horizon_steps < 1 or self.applied_actions_before_timeout < 1:
                raise ValueError("Motion timeout horizons must be positive.")

        @property
        def control_dt(self) -> float:
            """Applied-action period [s]."""
            return self.physics_dt * self.control_decimation

        @property
        def control_hz(self) -> float:
            """Applied-action rate [Hz]."""
            return 1.0 / self.control_dt

        @property
        def nominal_horizon_seconds(self) -> float:
            """Configured nominal episode duration [s]."""
            return self.configured_horizon_steps * self.control_dt

    @configclass
    class MocapAndFallResetCfg:
        """Motion-frame reset with a declared random-action fall subset."""

        motion_frame_probability: float = 0.8
        fall_probability: float = 0.2
        fall_random_actions_high_exclusive: int = 5

        def __post_init__(self) -> None:
            """Validate the reset mixture."""
            if not math.isclose(self.motion_frame_probability + self.fall_probability, 1.0):
                raise ValueError("Motion-frame and fall probabilities must sum to one.")
            if self.fall_random_actions_high_exclusive < 1:
                raise ValueError("Fall random-action upper bound must be positive.")

    @configclass
    class ReferenceResetCfg:
        """Reference-state reset with the native lie-down subset."""

        lie_down_probability: float = 0.3
        lie_down_root_height_m: float = 0.5
        """Root height used by lie-down resets [m]."""

        def __post_init__(self) -> None:
            """Validate reference-reset physical parameters."""
            if not 0.0 <= self.lie_down_probability <= 1.0:
                raise ValueError("lie_down_probability must be in [0, 1].")
            if not math.isfinite(self.lie_down_root_height_m) or self.lie_down_root_height_m <= 0.0:
                raise ValueError("lie_down_root_height_m must be finite and positive [m].")

    @configclass
    class HistoryCfg:
        """One fixed applied-transition history layout."""

        length: int = MISSING
        frame_width: int = MISSING
        layout: str = MISSING
        sources: tuple[str, ...] = MISSING
        include_reset_seed: bool = False

        def __post_init__(self) -> None:
            """Validate the compact history layout."""
            if self.length < 1 or self.frame_width < 1:
                raise ValueError("Motion history dimensions must be positive.")
            if not self.layout or not self.sources or len(set(self.sources)) != len(self.sources):
                raise ValueError("Motion history layout and sources must be nonempty and unique.")

        @property
        def width(self) -> int:
            """Flattened history width."""
            return self.length * self.frame_width

    @configclass
    class ObservationNoiseCfg:
        """Uniform actor noise declared before observation scaling."""

        enabled: bool = False
        uniform_half_ranges: dict[str, float] = {}
        privileged_enabled: bool = False

        def __post_init__(self) -> None:
            """Validate nonnegative finite uniform half-ranges."""
            if any(
                not name or not math.isfinite(value) or value < 0.0 for name, value in self.uniform_half_ranges.items()
            ):
                raise ValueError("Observation-noise half-ranges must be named, finite, and nonnegative.")
            if not self.enabled and self.uniform_half_ranges:
                raise ValueError("Disabled observation noise cannot declare half-ranges.")

    @configclass
    class RandomizationCfg:
        """Concrete startup, episodic, and push randomization ranges."""

        enabled: bool = False
        body_mass_scale_range: tuple[float, float] | None = None
        friction_range: tuple[float, float] | None = None
        torso_com_offset_range_m: tuple[float, float] | None = None
        """Per-axis torso center-of-mass offset range [m]."""

        default_joint_offset_range_rad: tuple[float, float] | None = None
        """Default joint-position offset range [rad]."""

        push_linear_velocity_range_m_s: tuple[float, float] | None = None
        """Horizontal push velocity range [m/s]."""

        push_angular_velocity_range_rad_s: tuple[float, float] | None = None
        """Angular push velocity range [rad/s]."""

        push_interval_seconds_integer_high_exclusive: tuple[int, int] | None = None

        def __post_init__(self) -> None:
            """Require ordered finite ranges only when randomization is active."""
            ranges = (
                self.body_mass_scale_range,
                self.friction_range,
                self.torso_com_offset_range_m,
                self.default_joint_offset_range_rad,
                self.push_linear_velocity_range_m_s,
                self.push_angular_velocity_range_rad_s,
            )
            if not self.enabled:
                if any(value is not None for value in ranges) or self.push_interval_seconds_integer_high_exclusive:
                    raise ValueError("Disabled randomization cannot declare ranges.")
                return
            if any(value is None for value in ranges) or self.push_interval_seconds_integer_high_exclusive is None:
                raise ValueError("Enabled randomization requires every frozen native range.")
            if any(
                not all(math.isfinite(component) for component in value) or value[0] > value[1]
                for value in ranges
                if value is not None
            ):
                raise ValueError("Motion randomization ranges must be finite and ordered.")
            low, high = self.push_interval_seconds_integer_high_exclusive
            if low < 1 or high <= low:
                raise ValueError("Push interval bounds must be positive and increasing [s].")

    @configclass
    class RouteCfg:
        """Named observation, history, and transition evidence routes."""

        transition_state_factory: Callable[[ManagerBasedRLEnv, MotionStatePayload], MotionRuntime] | str = MISSING
        behavior_action_width: int = MISSING
        actor_width: int = MISSING
        privileged_width: int = MISSING
        expert_width: int = MISSING
        forward_width: int = MISSING
        actor_fields: tuple[str, ...] = MISSING
        privileged_fields: tuple[str, ...] = MISSING
        expert_fields: tuple[str, ...] = MISSING
        forward_fields: tuple[str, ...] = MISSING
        raw_evidence: tuple[str, ...] = ()
        auxiliary_evidence: tuple[str, ...] = ()
        history: MotionProfileCfg.HistoryCfg | None = None

        def __post_init__(self) -> None:
            """Validate fixed route widths and unique named facts."""
            widths = (
                self.behavior_action_width,
                self.actor_width,
                self.privileged_width,
                self.expert_width,
                self.forward_width,
            )
            if any(value < 1 for value in widths):
                raise ValueError("Motion route widths must be positive.")
            if not callable(self.transition_state_factory) and not isinstance(self.transition_state_factory, str):
                raise TypeError("transition_state_factory must be callable or lazily resolvable.")
            for name, values in (
                ("actor_fields", self.actor_fields),
                ("privileged_fields", self.privileged_fields),
                ("expert_fields", self.expert_fields),
                ("forward_fields", self.forward_fields),
                ("raw_evidence", self.raw_evidence),
                ("auxiliary_evidence", self.auxiliary_evidence),
            ):
                if not values and name.endswith("_fields"):
                    raise ValueError(f"{name} must not be empty.")
                if len(set(values)) != len(values) or any(not value for value in values):
                    raise ValueError(f"{name} must contain unique nonempty names.")
            if not set(self.auxiliary_evidence).issubset(self.raw_evidence):
                raise ValueError("Auxiliary evidence must be selected from raw environment evidence.")

    identifier: str = MISSING
    timing: TimingCfg = MISSING
    reset: MocapAndFallResetCfg | ReferenceResetCfg = MISSING
    routes: RouteCfg = MISSING
    observation_noise: ObservationNoiseCfg = ObservationNoiseCfg()
    randomization: RandomizationCfg = RandomizationCfg()

    def __post_init__(self) -> None:
        """Validate the runtime profile identity."""
        if not self.identifier:
            raise ValueError("Motion profile identity must be nonempty.")


__all__ = ["MotionProfileCfg"]
