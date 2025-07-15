# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.recorders import PostStepStatesRecorderCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from .recorders import EndEffectorStateRecorder

##
# State recorders.
##


@configclass
class EndEffectorStateRecorderCfg(RecorderTermCfg):
    """Configuration for the end effector recorder term."""

    class_type: type[RecorderTerm] = EndEffectorStateRecorder


##
# Recorder manager configurations.
##


@configclass
class EndEffectorStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    record_post_step_states = PostStepStatesRecorderCfg()
    end_effector_states = EndEffectorStateRecorderCfg()
