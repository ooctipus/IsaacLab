# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class RslRlCommanderActorModelCfg(RslRlMLPModelCfg):
    """Configuration for the commander actor model.

    This extends :class:`RslRlMLPModelCfg` with parameters specific to
    :class:`CommanderActorModel` from rsl_rl. The commander network maps policy
    observations (excluding ``cmd_feat``) to a command feature vector that is
    injected into the observation dictionary before the actor MLP runs.
    """

    class_name: str = "CommanderActorModel"
    """The model class name. Defaults to CommanderActorModel."""

    commander_hidden_dims: list[int] = [256, 256, 256]
    """Hidden dimensions for the commander MLP."""

    commander_activation: str = "elu"
    """Activation function for the commander MLP. Defaults to elu."""

    commander_obs_normalization: bool = False
    """Whether to normalise commander inputs and targets. Defaults to False."""

    kinematic_reward_weight: float = 0.1
    """Scale applied to the kinematic tracking reward [1]. Defaults to 0.1."""

    commander_loss_coef: float = 0.1
    """Coefficient for the commander L2 regularisation loss [1]. Defaults to 0.1."""

    get_command_target_fn: Callable | None = None
    """Callable returning the ground-truth command target tensor.

    The callable should accept one argument (the environment) and return a tensor.
    """

    log_error_fn: Callable | None = None
    """Optional callable for logging command errors.

    The callable should accept ``(env, cmd_proposed, cmd_target)``.
    """


@configclass
class RslRlTaskEasingActorModelCfg(RslRlMLPModelCfg):
    """Configuration for the task-easing actor model.

    This extends :class:`RslRlMLPModelCfg` with parameters specific to
    :class:`TaskEasingActorModel` from rsl_rl. A chain of learned goal blocks
    progressively refines the task observation before passing it to the actor MLP.
    """

    class_name: str = "TaskEasingActorModel"
    """The model class name. Defaults to TaskEasingActorModel."""

    task_easing_constraint_fn: Literal["relu", "softplus"] = "relu"
    """Activation used as the constraint function. Defaults to relu."""

    task_easing_loss_coef: float = 0.0
    """Coefficient for the monotonic-value loss [1]. Defaults to 0.0."""

    task_easing_margin: float = 0.0
    """Margin in the monotonic-value constraint [1]. Defaults to 0.0."""

    task_easing_network: Literal["mlp", "residual"] = "mlp"
    """Backbone for goal blocks. Defaults to mlp."""

    num_goal_refinements: int = 2
    """Number of refinement steps. Defaults to 2."""

    goal_hidden_dims: list[int] = [256, 256]
    """Hidden dimensions for each goal block."""

    goal_activation: str | None = None
    """Activation for goal blocks. Defaults to the actor activation if not specified."""
