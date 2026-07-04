# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import warnings

warnings.warn(
    "scripts/reinforcement_learning/rsl_rl/play.py is deprecated. Use "
    "`./isaaclab.sh play --rl_library rsl_rl --task <TASK>` instead. "
    "Example: `./isaaclab.sh play --rl_library rsl_rl --task Isaac-Cartpole`.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in policy steps).")
parser.add_argument(
    "--video_physics_rate",
    action="store_true",
    default=False,
    help=(
        "Capture one frame per physics step instead of one per policy step. Produces a smoother video"
        " when decimation is high, at the cost of decimation× more renders during play."
    ),
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help=(
        "Sample actions from the policy distribution instead of using the deterministic mean."
        " Useful when the trained policy still has wide action variance (Policy/mean_std >> 0)"
        " and the deterministic mean alone is not goal-directed (sparse-reward PPO failure mode)."
    ),
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function. Both sides of this
# intersection are pre-fold (the callback reads the user's original sys.argv), so
# preset tokens like ``physics=NAME`` compare correctly here. Fold runs after.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.num_envs = env_cfg.scene.num_envs

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            # Per-seed sub-directory so concurrent renders with different seeds
            # don't clobber each other. Falls back to "play" when no seed is set.
            _video_subdir = f"seed_{args_cli.seed}" if args_cli.seed is not None else "play"
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", _video_subdir),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
                "fps": 20,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

            # Hack: capture 1 frame per physics step instead of 1 per policy step.
            # Default RecordVideo ticks once per env.step (= policy_dt), which looks
            # laggy when decimation is high. Hook ``sim.step`` to call
            # ``_capture_frame`` after each physics tick — ``_capture_frame`` itself
            # drives the render via ``env.render() → sim.render()``, producing one
            # fresh frame per tick. Disable the env's own conditional render
            # (``render_interval = inf``) so we don't double-render. Scale
            # ``video_length`` and ``frames_per_sec`` so the mp4 plays at physics rate.
            # Note: hooking ``sim.render`` would recurse — ``_capture_frame`` calls
            # ``env.render`` which calls ``sim.render``.
            #
            # ``ViewportCameraController._update_tracking_callback`` is bound to
            # Kit's ``post_update_event_stream``, which only ticks during the app
            # loop pump — not on every ``sim.render`` — so when ``origin_type`` is
            # ``asset_root``/``asset_body`` the camera lags the tracked asset by a
            # policy step. Force a camera-pose refresh right before each capture so
            # the render that ``_capture_frame`` triggers sees fresh asset coords.
            if args_cli.video_physics_rate:
                _decimation = env.unwrapped.cfg.decimation
                env.unwrapped.cfg.sim.render_interval = 10**9
                env.frames_per_sec = round(1.0 / env.unwrapped.physics_dt)
                env.video_length = args_cli.video_length * _decimation
                _record_wrapper = env
                _sim = env.unwrapped.sim
                _scene = env.unwrapped.scene
                _physics_dt = env.unwrapped.physics_dt
                _orig_sim_step = _sim.step
                _vcc = getattr(env.unwrapped, "viewport_camera_controller", None)

                # Unsubscribe Kit's ``post_update`` tracking callback. We refresh
                # the camera ourselves before each capture, and the Kit callback
                # races teardown — it fires after ``env.scene`` is deleted on exit
                # and raises ``AttributeError`` from a weakref proxy.
                if _vcc is not None:
                    _h = getattr(_vcc, "_viewport_camera_update_handle", None)
                    if _h is not None:
                        _h.unsubscribe()
                        _vcc._viewport_camera_update_handle = None

                def _refresh_viewer(_v=_vcc):
                    if _v is None:
                        return
                    origin = _v.cfg.origin_type
                    if origin == "asset_root" and _v.cfg.asset_name is not None:
                        _v.update_view_to_asset_root(_v.cfg.asset_name)
                    elif origin == "asset_body" and _v.cfg.asset_name is not None and _v.cfg.body_name is not None:
                        _v.update_view_to_asset_body(_v.cfg.asset_name, _v.cfg.body_name)

                def _sim_step_and_capture(
                    *args,
                    _o=_orig_sim_step,
                    _w=_record_wrapper,
                    _scn=_scene,
                    _dt=_physics_dt,
                    **kwargs,
                ):
                    out = _o(*args, **kwargs)
                    if _w.recording and len(_w.recorded_frames) < _w.video_length:
                        # Refresh scene buffers so ``asset.data.root_pos_w`` reflects
                        # the post-step state. The env loop calls ``scene.update``
                        # *after* ``sim.step``, so without this our viewer-refresh
                        # would read pre-step asset positions and the camera would
                        # track one physics tick behind, producing visible glitches.
                        _scn.update(dt=_dt)
                        _refresh_viewer()
                        _w._capture_frame()
                    return out

                _sim.step = _sim_step_and_capture

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner = agent_cfg.class_type(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            try:
                runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
                runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            except (RuntimeError, NotImplementedError):
                print("[INFO]: JIT/ONNX export not supported for this algorithm, skipping.")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt

        # reset environment
        obs = env.get_observations()
        timestep = 0
        # simulate environment
        try:
            while True:
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # agent stepping
                    if args_cli.stochastic:
                        actions = policy(obs, stochastic_output=True)
                    else:
                        actions = policy(obs)
                    # env stepping
                    obs, _, dones, _ = env.step(actions)
                    # reset recurrent states for episodes that have terminated
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)
                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            # close the simulator
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
