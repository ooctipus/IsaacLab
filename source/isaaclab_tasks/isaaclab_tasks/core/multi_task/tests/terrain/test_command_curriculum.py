# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock-based tests for the StateCommand <-> curriculum interplay.

The goal here is to exercise the MDP logic (task-table indexing, hold-time
bookkeeping, success-rate ring buffer, curriculum-driven resampling) without
running the IK pipeline that fills ``spawn_states``. Filling the spawn pool
takes tens of seconds of GPU time per test, so every test in this file builds
a synthetic ``TaskTable`` directly.

Organization:

* :class:`TestSuccessMonitor` - standalone, no env/robot required.
* :class:`TestCommandTerm` - indexing/hold-time invariants of
  :class:`StateCommand` with a mocked env + robot.
* :class:`TestCurriculum` - end-to-end curriculum + command-term loop,
  verifying the alias binding and ordering of success-update vs. resample.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab_tasks.core.multi_task.curriculum import (
    BetaSamplingStrategyCfg,
    Sampler,  # noqa: F401 -- used in fluent inline construction below,
    SamplerCfg,
    StateLayout,
    StateLayoutCfg,
    SuccessMonitor,
    SuccessMonitorCfg,
)
from isaaclab_tasks.core.multi_task.mdp.commands.state_command.state_command import StateCommand
from isaaclab_tasks.core.multi_task.mdp.curriculums import success_rate_sampler
from isaaclab_tasks.core.multi_task.terrain.mdp.commands.state_command_payloads import (
    CommandPayloadBaseFootState,
    CommandPayloadBaseState,
)
from isaaclab_tasks.core.multi_task.terrain.mdp.commands.task_table_builder import RelativeStateTaskTable
from isaaclab_tasks.core.multi_task.terrain.mdp.observations import (
    achieved_pos_env,
    command_current_state,
    command_target_state,
    target_pos_env,
)
from isaaclab_tasks.core.multi_task.utils.buffer_writers import FIFOBufferWriter


def _make_monitor(cfg: SuccessMonitorCfg) -> SuccessMonitor:
    """Construct a :class:`SuccessMonitor` with a fresh caller-owned rate tensor.

    The decoupled-rate-tensor interface requires the caller to allocate the
    success_rate tensor and pass it in. This helper centralizes that pattern
    for the tests in this file.
    """
    rate = torch.zeros(cfg.num_monitored_data, device=cfg.device)
    return SuccessMonitor(cfg, rate)


def _make_fifo_writer(
    num_streams: int,
    max_updates: int,
    use_warp: bool,
    start_ptr: torch.Tensor | None = None,
    size: torch.Tensor | None = None,
) -> FIFOBufferWriter:
    """Construct a FIFO writer with explicit caller-owned state tensors."""
    if start_ptr is None:
        start_ptr = torch.zeros(num_streams, device=DEVICE, dtype=torch.int32)
    if size is None:
        size = torch.zeros_like(start_ptr)
    changed_ids = torch.empty(max_updates, device=start_ptr.device, dtype=torch.int64)
    num_changed = torch.zeros(1, device=start_ptr.device, dtype=torch.int32)
    return FIFOBufferWriter(
        start_ptr=start_ptr,
        size=size,
        changed_ids=changed_ids,
        num_changed=num_changed,
        warp=use_warp,
    )


def _sample_by_target_rate(
    mon: SuccessMonitor,
    env_ids: torch.Tensor,
    target: float = 0.5,
    kappa: float = 2.0,
    return_probs: bool = False,
):
    """Build a one-shot Beta-only curriculum and sample from it.

    Used only by the legacy test cases in this file; production code
    constructs ``SamplerCfg`` once at init and reuses it across
    sample steps.
    """
    n = mon.success_rate.numel()
    layout = StateLayout(
        coords=torch.zeros(n, 1, device=mon.success_rate.device),
        spawn_index=torch.arange(n, device=mon.success_rate.device, dtype=torch.long),
    )
    curriculum = Sampler(
        SamplerCfg(
            strategies=[
                BetaSamplingStrategyCfg(target=target, kappa=kappa, weight=1.0, success_rate_bind="success_rates")
            ],
            eps=1e-8,
        ),
        layout,
        success_rates=mon.success_rate,
    )
    probs = curriculum.probabilities()
    choices = torch.multinomial(probs, len(env_ids), replacement=True).to(torch.int32)
    return (choices, probs) if return_probs else choices


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CURRICULUM_BINDS = {
    "success_rates_bind": "env.command_manager.get_term('goal_point').success_rates",
    "sample_indices_bind": "env.command_manager.get_term('goal_point').cmd_indices",
    "success_bind": "env.termination_manager.get_term('success')",
    "layout": StateLayoutCfg(
        coords_bind="env.command_manager.get_term('goal_point').table.spawn_states[:, :2]",
        spawn_index_bind="env.command_manager.get_term('goal_point').table.spawn_index",
        target_index_bind="env.command_manager.get_term('goal_point').table.target_index",
        task_partition_bind="env.command_manager.get_term('goal_point').table.task_partition",
    ),
}


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


# ---------------------------------------------------------------------------
# Synthetic TaskTable and mock env/robot helpers.
# ---------------------------------------------------------------------------


def _make_task_table(
    num_states: int = 8,
    num_joints: int = 12,
    pos_tasks: int = 10,
    pose_tasks: int = 10,
    device: str = DEVICE,
) -> RelativeStateTaskTable:
    """Build a synthetic TaskTable with two command kinds (pos + pose).

    ``spawn_states`` contain random, finite reset states
    (root_state + joint positions + joint velocities).
    ``params`` carry known position offsets so we can check target computation.
    ``task_mask`` differs between pos (first 3 cols) and pose (first 6 cols)
    so the mask handling is exercised.
    """
    gen = torch.Generator(device=device).manual_seed(0)

    # spawn_states: [num_states, 13 + 2 * num_joints]  (root_state, joint_pos, joint_vel)
    spawn_states = torch.zeros(num_states, 13 + 2 * num_joints, device=device)
    spawn_states[:, :3] = torch.randn(num_states, 3, generator=gen, device=device)
    spawn_states[:, 6] = 1.0  # identity quat (xyzw = 0,0,0,1)
    spawn_states[:, 13 : 13 + num_joints] = torch.randn(num_states, num_joints, generator=gen, device=device) * 0.1

    num_tasks = pos_tasks + pose_tasks
    spawn_index = torch.randint(0, num_states, (num_tasks,), generator=gen, device=device)
    target_index = torch.randint(0, num_states, (num_tasks,), generator=gen, device=device)
    tile_index = torch.randint(0, 4, (num_tasks,), generator=gen, device=device)

    # params: [num_tasks, 13] -- offsets for pos(0:3) + rot(3:6) + lin_vel(6:9) + ang_vel(9:12) + hold(12)
    params = torch.zeros(num_tasks, 13, device=device)
    params[:, :3] = torch.randn(num_tasks, 3, generator=gen, device=device) * 0.5
    params[:, 12] = 1.5  # hold time: 1.5s for every task

    # task_mask: [num_tasks, 12 + num_joints]
    task_mask = torch.zeros(num_tasks, 12 + num_joints, dtype=torch.bool, device=device)
    task_mask[:pos_tasks, :3] = True  # pos: x,y,z only
    task_mask[pos_tasks:, :6] = True  # pose: x,y,z,roll,pitch,yaw

    # CSR offsets and kind tags
    offsets = torch.tensor([0, pos_tasks, num_tasks], device=device, dtype=torch.long)
    task_partition = torch.bucketize(torch.arange(num_tasks, device=device), offsets[1:-1], right=True)
    kind = torch.tensor([0, 1], device=device, dtype=torch.long)  # 0=pos, 1=pose

    return RelativeStateTaskTable(
        num_tasks=num_tasks,
        spawn_index=spawn_index,
        target_index=target_index,
        tile_index=tile_index,
        params=params,
        task_mask=task_mask,
        payload_flags=torch.zeros(num_tasks, 1, dtype=torch.bool, device=device),
        offsets=offsets,
        task_partition=task_partition,
        kind=kind,
        spawn_states=spawn_states,
    )


class _MockRobot:
    """Minimal Articulation stand-in.

    Exposes warp-backed ``root_state_w``, ``root_quat_w``, ``joint_pos`` on
    ``.data`` and records every ``write_*_to_sim_index`` call.
    """

    def __init__(self, num_envs: int, num_joints: int, device: str):
        self.num_joints = num_joints
        self.device = device
        # Back root state with warp arrays so wp.to_torch works from _update_command.
        self._root_state_w = torch.zeros(num_envs, 13, device=device)
        self._root_state_w[:, 6] = 1.0  # identity quat (xyzw)
        self._root_quat_w = torch.zeros(num_envs, 4, device=device)
        self._root_quat_w[:, 3] = 1.0  # w=1
        self._joint_pos = torch.zeros(num_envs, num_joints, device=device)
        self._joint_vel = torch.zeros(num_envs, num_joints, device=device)
        self.body_names = ["base", "foot_0", "foot_1", "foot_2", "foot_3"]
        self._body_link_pos_w = torch.zeros(num_envs, len(self.body_names), 3, device=device)
        self.data = SimpleNamespace(
            root_state_w=wp.from_torch(self._root_state_w),
            root_quat_w=wp.from_torch(self._root_quat_w),
            joint_pos=wp.from_torch(self._joint_pos),
            joint_vel=wp.from_torch(self._joint_vel),
            body_link_pos_w=wp.from_torch(self._body_link_pos_w),
        )
        self.calls: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def find_bodies(self, names: str | list[str], preserve_order: bool = False):
        if isinstance(names, str):
            names = [names]
        body_ids = [self.body_names.index(name) for name in names]
        return body_ids, [self.body_names[body_id] for body_id in body_ids]

    def write_root_link_pose_to_sim_index(self, root_pose: torch.Tensor, env_ids: torch.Tensor):
        self._root_state_w[env_ids, :7] = root_pose
        self._root_quat_w[env_ids] = root_pose[:, 3:7]
        self.calls.append(("root_pose", env_ids.clone(), root_pose.clone()))

    def write_root_com_velocity_to_sim_index(self, root_velocity: torch.Tensor, env_ids: torch.Tensor):
        self._root_state_w[env_ids, 7:] = root_velocity
        self.calls.append(("root_velocity", env_ids.clone(), root_velocity.clone()))

    def write_joint_state_to_sim_index(self, position: torch.Tensor, velocity: torch.Tensor, env_ids: torch.Tensor):
        self._joint_pos[env_ids] = position
        self._joint_vel[env_ids] = velocity
        self.calls.append(("joint_state", env_ids.clone(), torch.cat([position, velocity], dim=-1).clone()))

    def write_root_pose_to_sim_index(self, root_pose: torch.Tensor, env_ids: torch.Tensor):
        self._root_state_w[env_ids, :7] = root_pose
        self.calls.append(("pose", env_ids.clone(), root_pose.clone()))

    def write_root_velocity_to_sim_index(self, root_velocity: torch.Tensor, env_ids: torch.Tensor):
        self._root_state_w[env_ids, 7:13] = root_velocity
        self.calls.append(("vel", env_ids.clone(), root_velocity.clone()))

    def write_joint_position_to_sim_index(self, position: torch.Tensor, env_ids: torch.Tensor):
        self._joint_pos[env_ids] = position
        self.calls.append(("jpos", env_ids.clone(), position.clone()))

    def write_joint_velocity_to_sim_index(self, velocity: torch.Tensor, env_ids: torch.Tensor):
        self.calls.append(("jvel", env_ids.clone(), velocity.clone()))


class _MockScene(SimpleNamespace):
    def __getitem__(self, name: str):
        return self._articulations[name]


def _make_env(num_envs: int, device: str, step_dt: float = 0.02):
    """Build a minimal env namespace with just the attributes touched by the MDP code under test."""
    return SimpleNamespace(
        num_envs=num_envs,
        device=device,
        step_dt=step_dt,
        common_step_counter=0,
        extras={"log": {}},
        termination_manager=None,
        command_manager=None,
        scene=None,
    )


def _make_command_term(
    env,
    table: RelativeStateTaskTable,
    num_joints: int = 12,
    payload_class: type = CommandPayloadBaseFootState,
    device: str = DEVICE,
) -> StateCommand:
    """Construct a StateCommand without invoking its __init__.

    The real __init__ calls build_task_table (the expensive IK pipeline) and
    requires an actual terrain + asset, neither of which we want in a mock
    test. Instead we allocate the object, set only the fields the methods
    under test reach for, and return it ready to use.
    """
    term = object.__new__(StateCommand)
    term._env = env
    cmd_names = {"pos_cmd": None, "pose_cmd": None}
    # A thin cfg only needs .commands (used in _update_metrics) and std scales.
    term.cfg = SimpleNamespace(
        commands=cmd_names,
        resampling_time_range=(1.0, 1.0),
        randomize_command_indices=True,
        debug_vis=False,
        task_table=SimpleNamespace(pipeline_cfg=SimpleNamespace(asset_cfg=SimpleNamespace(name="robot"))),
        payload=SimpleNamespace(
            class_type=payload_class,
            pos_std=0.5,
            rot_std=0.5,
            lin_vel_std=0.5,
            ang_vel_std=0.5,
            foot_pos_std=0.1,
            normalize_command_obs=False,
        ),
    )
    term.robot = _MockRobot(env.num_envs, num_joints, device)
    env.scene = _MockScene(
        _articulations={"robot": term.robot},
        _rigid_objects={},
        env_origins=torch.zeros(env.num_envs, 3, device=device),
        _default_env_origins=torch.zeros(env.num_envs, 3, device=device),
        env_ns="/World/envs",
        env_regex_ns="/World/envs/env_.*",
        terrain=SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/ground")),
    )
    term._reset_assets = ["robot"]
    term.table = table
    term.success_rates = torch.zeros(table.num_tasks, device=device, dtype=torch.float32)

    foot_body_ids = [1, 2, 3, 4]
    newton_foot_body_ids = foot_body_ids
    isaac_to_newton_joint_order = torch.arange(num_joints, device=device, dtype=torch.long)

    class _FakeTargetFk:
        model = SimpleNamespace(
            joint_coord_count=7 + num_joints,
            joint_dof_count=num_joints,
            body_count=len(foot_body_ids) + 1,
        )

        def eval_fk_batched(self, joint_q, joint_qd, body_q, body_qd):
            wp.to_torch(body_q).zero_()

    table.target_fk_kin = _FakeTargetFk()
    table.newton_foot_body_ids = newton_foot_body_ids
    table.isaac_to_newton_joint_order = isaac_to_newton_joint_order
    table.foot_body_ids = foot_body_ids
    term._payload = payload_class(term.cfg, env, table)

    # cmd_buf / cmd_mask / cmd_ids are now owned + allocated by the payload
    term.states_relative = False
    term.cmd_indices = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    term.randomize_command_indices = True
    term._command = torch.zeros(env.num_envs, term._payload.command_dim, device=device)
    term._err = torch.empty(env.num_envs, term._payload.error_dim, device=device)
    term.metrics = {name: torch.zeros(env.num_envs, device=device) for name in term._payload.error_names}
    term.metrics["instant_success"] = torch.zeros(env.num_envs, device=device)
    term._debug_vis_handle = None  # CommandTerm.__del__ expects this attribute
    return term


# ---------------------------------------------------------------------------
# SuccessMonitor: ring-buffer + sampler properties.
# ---------------------------------------------------------------------------


class TestSuccessMonitor:
    """Ring-buffer correctness and sampling distribution of :class:`SuccessMonitor`."""

    def test_rate_updates_in_place_alias_preserved(self):
        """success_rate is updated via ``[:]`` so external aliases must survive."""
        cfg = SuccessMonitorCfg(monitored_history_len=8, num_monitored_data=4, device=DEVICE)
        mon = _make_monitor(cfg)
        alias = mon.success_rate  # external reference (what the curriculum binds)

        ids = torch.tensor([0, 1, 2, 3], device=DEVICE)
        mask = torch.tensor([True, True, False, False], device=DEVICE)
        mon.success_update(ids, mask)

        # same tensor object after update
        assert mon.success_rate.data_ptr() == alias.data_ptr()
        torch.testing.assert_close(alias[:2], torch.ones(2, device=DEVICE))
        torch.testing.assert_close(alias[2:], torch.zeros(2, device=DEVICE))

    def test_ring_buffer_wraps_at_history_len(self):
        """Success rate for id=0 tracks the last ``monitored_history_len`` entries only."""
        hist = 4
        cfg = SuccessMonitorCfg(monitored_history_len=hist, num_monitored_data=2, device=DEVICE)
        mon = _make_monitor(cfg)

        # 4 successes -> rate = 1.0
        for _ in range(hist):
            mon.success_update(
                torch.tensor([0], device=DEVICE),
                torch.tensor([True], device=DEVICE),
            )
        assert float(mon.success_rate[0].item()) == pytest.approx(1.0)
        assert int(mon.success_size[0].item()) == hist

        # now 4 failures: ring-buffer should roll over completely
        for _ in range(hist):
            mon.success_update(
                torch.tensor([0], device=DEVICE),
                torch.tensor([False], device=DEVICE),
            )
        assert float(mon.success_rate[0].item()) == pytest.approx(0.0)

        # id=1 was never written: still zero rate, zero size.
        assert int(mon.success_size[1].item()) == 0

    def test_batched_update_counts_duplicates(self):
        """A single call with repeated ids must record every event for that id."""
        cfg = SuccessMonitorCfg(monitored_history_len=16, num_monitored_data=3, device=DEVICE)
        mon = _make_monitor(cfg)

        ids = torch.tensor([0, 0, 0, 1, 1, 2], device=DEVICE)
        mask = torch.tensor([True, True, False, True, False, True], device=DEVICE)
        mon.success_update(ids, mask)

        assert int(mon.success_size[0].item()) == 3
        assert int(mon.success_size[1].item()) == 2
        assert int(mon.success_size[2].item()) == 1
        torch.testing.assert_close(mon.success_count, torch.tensor([2, 1, 1], device=DEVICE, dtype=torch.int32))
        assert float(mon.success_rate[0].item()) == pytest.approx(2.0 / 3.0)
        assert float(mon.success_rate[1].item()) == pytest.approx(0.5)
        assert float(mon.success_rate[2].item()) == pytest.approx(1.0)

    def test_single_stream_burst_keeps_tail(self):
        """A large same-id update should keep only the newest history window."""
        hist = 4
        cfg = SuccessMonitorCfg(monitored_history_len=hist, num_monitored_data=2, device=DEVICE)
        mon = _make_monitor(cfg)

        ids = torch.zeros(10, dtype=torch.long, device=DEVICE)
        mask = torch.tensor([True, True, True, False, False, False, True, False, True, False], device=DEVICE)
        mon.success_update(ids, mask)

        assert int(mon.success_size[0].item()) == hist
        assert int(mon.success_pointer[0].item()) == 2
        assert int(mon.success_count[0].item()) == 2
        assert float(mon.success_rate[0].item()) == pytest.approx(0.5)

    def test_success_update_requires_bool_mask(self):
        """SuccessMonitor owns the bool outcome contract."""
        for use_warp in (False, True):
            cfg = SuccessMonitorCfg(
                monitored_history_len=4,
                num_monitored_data=2,
                device=DEVICE,
                max_updates=1,
                warp=use_warp,
            )
            mon = _make_monitor(cfg)

            with pytest.raises(TypeError, match="success_mask"):
                mon.success_update(
                    torch.tensor([0], device=DEVICE, dtype=torch.int64),
                    torch.tensor([1.0], device=DEVICE),
                )

    def test_fifo_buffer_writer_uses_external_state_tensors(self):
        """FIFOBufferWriter should write through explicit external state tensors."""
        start_ptr = torch.tensor([1, 0], device=DEVICE, dtype=torch.int32)
        size = torch.tensor([1, 0], device=DEVICE, dtype=torch.int32)
        for use_warp in (False, True):
            data = torch.zeros(2, 3, device=DEVICE, dtype=torch.bool)
            buffer_writer = _make_fifo_writer(2, 4, use_warp, start_ptr.clone(), size.clone())

            buffer_writer.add(
                data,
                torch.tensor([0, 0, 0, 0], device=DEVICE, dtype=torch.int64),
                torch.tensor([False, True, True, False], device=DEVICE),
            )
            if use_warp and torch.cuda.is_available():
                torch.cuda.synchronize()

            assert int(buffer_writer.num_changed[0].item()) == 1
            torch.testing.assert_close(buffer_writer.changed_ids[:1], torch.tensor([0], device=DEVICE))
            assert int(buffer_writer.start_ptr[0].item()) == 2
            assert int(buffer_writer.size[0].item()) == 3
            torch.testing.assert_close(data[0], torch.tensor([True, False, True], device=DEVICE))

    def test_fifo_buffer_writer_add_groups_raw_ids(self):
        """Raw ``add`` should group duplicate ids while preserving per-stream FIFO order."""
        for use_warp in (False, True):
            data = torch.zeros(3, 4, device=DEVICE, dtype=torch.bool)
            buffer_writer = _make_fifo_writer(3, 5, use_warp)

            buffer_writer.add(
                data,
                torch.tensor([2, 0, 2, 2, 0], device=DEVICE, dtype=torch.int64),
                torch.tensor([True, False, False, True, True], device=DEVICE),
            )
            if use_warp and torch.cuda.is_available():
                torch.cuda.synchronize()

            assert int(buffer_writer.num_changed[0].item()) == 2
            torch.testing.assert_close(buffer_writer.changed_ids[:2], torch.tensor([0, 2], device=DEVICE))
            torch.testing.assert_close(data[0], torch.tensor([False, True, False, False], device=DEVICE))
            torch.testing.assert_close(data[2], torch.tensor([True, False, True, False], device=DEVICE))
            torch.testing.assert_close(
                buffer_writer.start_ptr, torch.tensor([2, 0, 3], device=DEVICE, dtype=torch.int32)
            )
            torch.testing.assert_close(buffer_writer.size, torch.tensor([2, 0, 3], device=DEVICE, dtype=torch.int32))

    def test_fifo_buffer_writer_warp_add_supports_non_bool_payloads(self):
        """Warp FIFO writes should support non-bool dtype and trailing payload dimensions."""
        data = torch.zeros(3, 4, 2, device=DEVICE, dtype=torch.float32)
        buffer_writer = _make_fifo_writer(3, 5, use_warp=True)
        values = torch.tensor([[20.0, 21.0], [0.0, 1.0], [22.0, 23.0], [24.0, 25.0], [2.0, 3.0]], device=DEVICE)

        buffer_writer.add(data, torch.tensor([2, 0, 2, 2, 0], device=DEVICE, dtype=torch.int64), values)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        torch.testing.assert_close(
            data[0],
            torch.tensor([[0.0, 1.0], [2.0, 3.0], [0.0, 0.0], [0.0, 0.0]], device=DEVICE),
        )
        torch.testing.assert_close(
            data[2],
            torch.tensor([[20.0, 21.0], [22.0, 23.0], [24.0, 25.0], [0.0, 0.0]], device=DEVICE),
        )
        torch.testing.assert_close(buffer_writer.start_ptr, torch.tensor([2, 0, 3], device=DEVICE, dtype=torch.int32))
        torch.testing.assert_close(buffer_writer.size, torch.tensor([2, 0, 3], device=DEVICE, dtype=torch.int32))

    def test_success_monitor_warp_update_matches_torch_path(self):
        """Warp monitor updates should match the eager Torch success-rate path."""
        ids = torch.tensor([2, 0, 2, 2, 0], device=DEVICE, dtype=torch.int64)
        mask = torch.tensor([True, False, False, True, True], device=DEVICE)
        torch_cfg = SuccessMonitorCfg(monitored_history_len=4, num_monitored_data=3, device=DEVICE)
        warp_cfg = SuccessMonitorCfg(
            monitored_history_len=4,
            num_monitored_data=3,
            device=DEVICE,
            max_updates=ids.numel(),
            warp=True,
        )
        torch_mon = _make_monitor(torch_cfg)
        warp_mon = _make_monitor(warp_cfg)

        torch_mon.success_update(ids, mask)
        warp_mon.success_update(ids, mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        torch.testing.assert_close(warp_mon.success_buf, torch_mon.success_buf)
        torch.testing.assert_close(warp_mon.success_size, torch_mon.success_size)
        torch.testing.assert_close(warp_mon.success_pointer, torch_mon.success_pointer)
        torch.testing.assert_close(warp_mon.success_count, torch_mon.success_count)
        torch.testing.assert_close(warp_mon.success_rate, torch_mon.success_rate)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA.")
    def test_success_monitor_warp_update_captures(self):
        """Warp monitor update should be capturable with raw ids and success values."""
        cfg = SuccessMonitorCfg(
            monitored_history_len=4,
            num_monitored_data=3,
            device="cuda:0",
            max_updates=5,
            warp=True,
        )
        mon = _make_monitor(cfg)
        ids = torch.tensor([2, 0, 2, 2, 0], device="cuda:0", dtype=torch.int64)
        mask = torch.tensor([True, False, False, True, True], device="cuda:0")

        # Warm compile before capture.
        mon.success_update(ids, mask)
        torch.cuda.synchronize()
        mon.success_buf.zero_()
        mon.success_pointer.zero_()
        mon.success_size.zero_()
        mon.success_count.zero_()
        mon.success_rate.zero_()
        torch.cuda.synchronize()

        wp.capture_begin(device="cuda:0")
        mon.success_update(ids, mask)
        graph = wp.capture_end(device="cuda:0")
        wp.capture_launch(graph)
        torch.cuda.synchronize()

        torch.testing.assert_close(mon.success_rate, torch.tensor([0.5, 0.0, 2.0 / 3.0], device="cuda:0"))

    def test_sample_by_target_rate_matches_mode(self):
        """Sampling with ``target`` near a bin concentrates choices there.

        We build 4 populations with distinct success rates, then check that
        ``target=0`` prefers the lowest bin and ``target=1`` prefers the highest.
        """
        torch.manual_seed(0)
        hist = 10
        cfg = SuccessMonitorCfg(monitored_history_len=hist, num_monitored_data=4, device=DEVICE)
        mon = _make_monitor(cfg)
        for bin_id in range(4):
            n_ok = bin_id * 3  # 0, 3, 6, 9 successes out of 10
            ids = torch.full((hist,), bin_id, dtype=torch.long, device=DEVICE)
            mask = torch.zeros(hist, dtype=torch.bool, device=DEVICE)
            mask[:n_ok] = True
            mon.success_update(ids, mask)

        env_ids = torch.arange(4096, device=DEVICE)
        low = _sample_by_target_rate(mon, env_ids, target=0.0, kappa=5.0)
        high = _sample_by_target_rate(mon, env_ids, target=1.0, kappa=5.0)
        mid = _sample_by_target_rate(mon, env_ids, target=0.5, kappa=5.0)

        low_counts = torch.bincount(low.long(), minlength=4).float() / env_ids.numel()
        high_counts = torch.bincount(high.long(), minlength=4).float() / env_ids.numel()
        mid_counts = torch.bincount(mid.long(), minlength=4).float() / env_ids.numel()

        # target=0 -> most mass on bin 0 (the 0%-rate population)
        assert torch.argmax(low_counts).item() == 0
        # target=1 -> most mass on bin 3 (the 90%-rate population)
        assert torch.argmax(high_counts).item() == 3
        # target=0.5 -> most mass on an interior bin (1 or 2), not the extremes
        assert torch.argmax(mid_counts).item() in (1, 2)

    def test_sample_returns_probs_that_sum_to_one(self):
        """``return_probs=True`` must give a valid distribution over partitions."""
        cfg = SuccessMonitorCfg(monitored_history_len=4, num_monitored_data=5, device=DEVICE)
        mon = _make_monitor(cfg)
        env_ids = torch.arange(3, device=DEVICE)
        _, probs = _sample_by_target_rate(mon, env_ids, target=0.5, return_probs=True)
        assert probs.shape == (5,)
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0, device=DEVICE))
        assert (probs >= 0).all()


# ---------------------------------------------------------------------------
# StateCommand: MDP accuracy with the mock environment.
# ---------------------------------------------------------------------------


class TestCommandTerm:
    """Exercise the command term in isolation -- no curriculum, no IK pipeline."""

    def test_cmd_ids_are_derived_from_selected_task_rows(self):
        """``cmd_ids`` must match the CSR bucket that ``cmd_indices`` lands in."""
        torch.manual_seed(0)
        table = _make_task_table()
        env = _make_env(num_envs=64, device=DEVICE)
        term = _make_command_term(env, table)

        env_ids = torch.arange(env.num_envs, device=DEVICE)
        # cmd_ids is stored by the payload at resample time from the selected rows
        term.randomize_command_indices = False
        term.cmd_indices[env_ids] = torch.randint(0, table.num_tasks, (env.num_envs,), device=DEVICE)
        term._resample_command(env_ids)

        idx = term.cmd_indices[env_ids]
        ids = term._payload.cmd_ids[env_ids]
        offsets = table.offsets
        lo = offsets[ids.long()]
        hi = offsets[ids.long() + 1]
        assert ((idx >= lo) & (idx < hi)).all(), "cmd_id bucket does not contain cmd_indices"

    def test_resample_command_populates_target_and_teleports(self):
        """After ``_resample_command``:
        * target pos = spawn_states[target_idx, :3] + params[task_idx, :3]
        * hold column = params[task_idx, 12]
        * robot received root pose and COM velocity writes from spawn_states[spawn_idx, :13]
        * cmd_mask equals table.task_mask at task_idx
        """
        torch.manual_seed(123)
        table = _make_task_table()
        env = _make_env(num_envs=16, device=DEVICE)
        term = _make_command_term(env, table)

        env_ids = torch.arange(env.num_envs, device=DEVICE)
        term._resample_command(env_ids)

        task_idx = term.cmd_indices[env_ids]
        target_state_idx = table.target_index[task_idx]
        spawn_state_idx = table.spawn_index[task_idx]

        expected_target_pos = table.spawn_states[target_state_idx, :3] + table.params[task_idx, :3]
        torch.testing.assert_close(term._payload.cmd_buf[env_ids, 0, :3], expected_target_pos)
        torch.testing.assert_close(term._payload.cmd_buf[env_ids, 0, 3:12], table.params[task_idx, 3:12])
        torch.testing.assert_close(
            term._payload.cmd_buf[env_ids, 0, 12 : 12 + term._payload.num_joints],
            table.spawn_states[target_state_idx, 13 : 13 + term._payload.num_joints],
        )
        torch.testing.assert_close(
            term._payload.cmd_buf[env_ids, 0, term._payload.time_idx], table.params[task_idx, 12]
        )
        torch.testing.assert_close(
            term._payload.cmd_buf[env_ids, 1, term._payload.time_idx], table.params[task_idx, 12]
        )
        torch.testing.assert_close(term.command[:, :12], term._payload.cmd_buf[:, 1, :12])
        assert not term.get_task_done().any()
        assert (term._payload.cmd_mask[env_ids] == table.task_mask[task_idx]).all()

        # Robot was teleported to the spawn reset state associated with each task.
        root_pose_calls = [c for c in term.robot.calls if c[0] == "root_pose"]
        root_velocity_calls = [c for c in term.robot.calls if c[0] == "root_velocity"]
        assert len(root_pose_calls) == len(root_velocity_calls) == 1
        _, call_env_ids, root_pose = root_pose_calls[0]
        _, velocity_env_ids, root_velocity = root_velocity_calls[0]
        torch.testing.assert_close(call_env_ids, env_ids)
        torch.testing.assert_close(velocity_env_ids, env_ids)
        torch.testing.assert_close(root_pose, table.spawn_states[spawn_state_idx, :7])
        torch.testing.assert_close(root_velocity, table.spawn_states[spawn_state_idx, 7:13])

    def test_spawn_states_are_not_shifted_by_env_origins(self):
        """Task-table states are already valid world terrain poses."""
        table = _make_task_table()
        env = _make_env(num_envs=2, device=DEVICE)
        term = _make_command_term(env, table)
        env.scene.env_origins[:] = torch.tensor([[10.0, 20.0, 0.0], [-5.0, 2.0, 0.0]], device=DEVICE)

        env_ids = torch.arange(env.num_envs, device=DEVICE)
        term.cmd_indices[env_ids] = torch.tensor([0, 1], device=DEVICE)
        term._resample_command(env_ids)

        task_idx = term.cmd_indices[env_ids]
        spawn_state_idx = table.spawn_index[task_idx]

        root_pose_calls = [c for c in term.robot.calls if c[0] == "root_pose"]
        _, _, root_pose = root_pose_calls[-1]
        torch.testing.assert_close(root_pose[:, :3], table.spawn_states[spawn_state_idx, :3])

    def test_replicated_terrain_spawn_states_are_shifted_by_env_origins(self):
        """Replicated terrain task-table states are placed into each env world slot."""
        table = _make_task_table()
        env = _make_env(num_envs=2, device=DEVICE)
        term = _make_command_term(env, table)
        origins = torch.tensor([[10.0, 20.0, 0.0], [-5.0, 2.0, 0.0]], device=DEVICE)
        env.scene.env_origins[:] = origins
        # replicated terrain: the env declares its stored states as env-local, so
        # the command lifts spawn/target by env_origins (no terrain sniffing)
        term.states_relative = True

        env_ids = torch.arange(env.num_envs, device=DEVICE)
        term.cmd_indices[env_ids] = torch.tensor([0, 1], device=DEVICE)
        term._resample_command(env_ids)

        task_idx = term.cmd_indices[env_ids]
        spawn_state_idx = table.spawn_index[task_idx]

        root_pose_calls = [c for c in term.robot.calls if c[0] == "root_pose"]
        _, _, root_pose = root_pose_calls[-1]
        torch.testing.assert_close(root_pose[:, :3], table.spawn_states[spawn_state_idx, :3] + origins)

    def test_terrain_task_target_uses_valid_target_state(self):
        """Terrain commands should target the sampled IK-valid state, not random pose params."""
        table = _make_task_table(pos_tasks=1, pose_tasks=1)
        env = _make_env(num_envs=1, device=DEVICE)
        term = _make_command_term(env, table)

        task_id = torch.tensor([0], dtype=torch.long, device=DEVICE)
        target_state_id = int(table.target_index[0].item())
        yaw = torch.tensor(1.0, device=DEVICE)
        table.spawn_states[target_state_id : target_state_id + 1, :3] = torch.tensor([[1.0, 2.0, 3.0]], device=DEVICE)
        table.spawn_states[target_state_id : target_state_id + 1, 3:7] = torch.tensor(
            [[0.0, 0.0, torch.sin(yaw * 0.5), torch.cos(yaw * 0.5)]], device=DEVICE
        )
        table.spawn_states[target_state_id : target_state_id + 1, 13 : 13 + term._payload.num_joints] = 0.25
        target_state = table.spawn_states[target_state_id]

        table.payload_flags[task_id, 0] = True
        table.task_mask[task_id, :6] = True
        table.task_mask[task_id, 12:] = True
        table.params[task_id, :6] = torch.tensor([[9.0, 9.0, 9.0, 0.3, 0.4, 0.5]], device=DEVICE)
        table.params[task_id, 12] = 0.75

        term.cmd_indices[:] = task_id
        term.randomize_command_indices = False
        term._resample_command(torch.tensor([0], device=DEVICE, dtype=torch.long))

        torch.testing.assert_close(term._payload.cmd_buf[0, 0, :3], target_state[:3])
        torch.testing.assert_close(term._payload.cmd_buf[0, 0, 3:6], torch.tensor([0.0, 0.0, 1.0], device=DEVICE))
        torch.testing.assert_close(
            term._payload.cmd_buf[0, 0, 12 : 12 + term._payload.num_joints],
            target_state[13 : 13 + term._payload.num_joints],
        )
        assert term._payload.cmd_buf[0, 0, term._payload.time_idx].item() == pytest.approx(0.75)
        assert bool(term._payload.foot_success_mask[0])

    def test_base_state_payload_keeps_terrain_target_without_feet(self):
        """Base-state payload uses terrain target rows but has no foot command channels."""
        table = _make_task_table(pos_tasks=1, pose_tasks=1)
        env = _make_env(num_envs=1, device=DEVICE)
        term = _make_command_term(env, table, payload_class=CommandPayloadBaseState)

        task_id = torch.tensor([0], dtype=torch.long, device=DEVICE)
        target_state_id = int(table.target_index[0].item())
        table.spawn_states[target_state_id : target_state_id + 1, :3] = torch.tensor([[1.0, 2.0, 3.0]], device=DEVICE)
        table.spawn_states[target_state_id : target_state_id + 1, 3:7] = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], device=DEVICE
        )
        table.payload_flags[task_id, 0] = True
        table.task_mask[task_id, :6] = True
        table.params[task_id, 12] = 0.75

        term.cmd_indices[:] = task_id
        term.randomize_command_indices = False
        term._resample_command(torch.tensor([0], device=DEVICE, dtype=torch.long))

        torch.testing.assert_close(term._payload.cmd_buf[0, 0, :3], table.spawn_states[target_state_id, :3])
        assert term.command.shape == (env.num_envs, 12)
        assert not hasattr(term._payload, "target_foot_pos_w")
        assert not hasattr(term._payload, "foot_success_mask")

    def test_hold_time_drains_only_when_all_groups_pass(self):
        """``current[ti]`` advances by ``step_dt`` iff all 5 error groups are below threshold.

        We set up two envs:
          * env 0: current == target everywhere -> all groups pass, should tick
          * env 1: target pos offset by a large value -> pos group fails, no tick
        After ``num_ticks`` updates, env 0's hold should have drained by
        ``num_ticks * step_dt`` and env 1's should be unchanged.
        """
        table = _make_task_table()
        env = _make_env(num_envs=2, device=DEVICE, step_dt=0.1)
        term = _make_command_term(env, table)

        # Force a known task for both envs: task 0 (pos-only, pos mask on x,y,z).
        term.cmd_indices[:] = 0
        term._payload.cmd_mask[:] = table.task_mask[0]

        # Seed cmd_buf: target = arbitrary, current = target (aligned); hold = 1.0.
        hold_init = 1.0
        target = torch.tensor([0.5, -0.3, 0.8], device=DEVICE)
        term._payload.cmd_buf[:, 0, :3] = target
        term._payload.cmd_buf[:, 0, term._payload.time_idx] = hold_init
        term._payload.cmd_buf[:, 2, term._payload.time_idx] = 0.0

        # env 0: robot at target (error=0). env 1: robot offset by +5m in x.
        rs = term.robot._root_state_w
        rs[0, :3] = target
        rs[1, :3] = target + torch.tensor([5.0, 0.0, 0.0], device=DEVICE)
        rs[:, 6] = 1.0  # identity quat xyzw (w already 1, others 0)
        rs[:, 3:6] = 0.0
        term.robot._root_quat_w[:, :3] = 0.0
        term.robot._root_quat_w[:, 3] = 1.0

        num_ticks = 5
        for _ in range(num_ticks):
            term._update_command()

        # env 0 accumulated num_ticks * step_dt successes.
        expected_current = num_ticks * env.step_dt
        assert term._payload.cmd_buf[0, 2, term._payload.time_idx].item() == pytest.approx(expected_current)
        # env 0's delta[time_idx] = target - current = hold_init - expected
        assert term._payload.cmd_buf[0, 1, term._payload.time_idx].item() == pytest.approx(hold_init - expected_current)
        # env 1 never ticked
        assert term._payload.cmd_buf[1, 2, term._payload.time_idx].item() == pytest.approx(0.0)
        assert term._payload.cmd_buf[1, 1, term._payload.time_idx].item() == pytest.approx(hold_init)

    def test_command_observation_exposes_target_feet_not_joint_delta(self):
        """Policy command is root delta plus target foot positions in base frame."""
        table = _make_task_table()
        env = _make_env(num_envs=2, device=DEVICE)
        term = _make_command_term(env, table)

        root_pos = torch.tensor([[1.0, 2.0, 0.5], [-1.0, 0.5, 0.25]], device=DEVICE)
        target_feet = torch.tensor(
            [
                [[1.3, 2.0, 0.1], [1.0, 2.2, 0.1], [0.8, 1.9, 0.1], [1.2, 1.7, 0.1]],
                [[-0.7, 0.5, 0.0], [-1.0, 0.7, 0.0], [-1.2, 0.4, 0.0], [-0.8, 0.2, 0.0]],
            ],
            device=DEVICE,
        )
        term.robot._root_state_w[:, :3] = root_pos
        term.robot._root_state_w[:, 3:6] = 0.0
        term.robot._root_state_w[:, 6] = 1.0
        term.robot._root_quat_w[:, :3] = 0.0
        term.robot._root_quat_w[:, 3] = 1.0
        term._payload.target_foot_pos_w.copy_(target_feet)
        term._payload.foot_success_mask[:] = torch.tensor([True, False], device=DEVICE)

        term._update_command()

        num_feet = term._payload.num_feet
        assert term.command.shape == (env.num_envs, 12 + 3 * num_feet)
        torch.testing.assert_close(term.command[0, 12:].view(num_feet, 3), target_feet[0] - root_pos[0])
        torch.testing.assert_close(term.command[1, 12:], torch.zeros(3 * num_feet, device=DEVICE))
        torch.testing.assert_close(
            term._payload.cmd_buf[:, 1, 12 : 12 + term._payload.num_joints],
            torch.zeros_like(term.robot._joint_pos),
        )

    def test_state_observations_use_command_boundary(self):
        """CRL state observations read payload-defined state through the command term."""
        table = _make_task_table()
        env = _make_env(num_envs=2, device=DEVICE)
        term = _make_command_term(env, table)
        env.command_manager = SimpleNamespace(get_term=lambda name: term)
        env.scene.terrain = SimpleNamespace(
            env_origins=torch.tensor([[10.0, 20.0, 0.0], [-3.0, 4.0, 1.0]], device=DEVICE)
        )

        current_root = torch.arange(24, device=DEVICE, dtype=torch.float32).view(2, 12)
        target_root = current_root + 100.0
        term._payload.cmd_buf[:, 2, :12] = current_root
        term._payload.cmd_buf[:, 0, :12] = target_root
        term._payload.current_foot_pos_w.copy_(torch.arange(24, device=DEVICE, dtype=torch.float32).view(2, 4, 3))
        term._payload.target_foot_pos_w.copy_(term._payload.current_foot_pos_w + 50.0)

        origins = env.scene.terrain.env_origins
        current_expected = torch.cat(
            [
                current_root[:, :3] - origins,
                current_root[:, 3:12],
                (term._payload.current_foot_pos_w - origins[:, None]).flatten(1),
            ],
            dim=-1,
        )
        target_expected = torch.cat(
            [
                target_root[:, :3] - origins,
                target_root[:, 3:12],
                (term._payload.target_foot_pos_w - origins[:, None]).flatten(1),
            ],
            dim=-1,
        )
        torch.testing.assert_close(command_current_state(env), current_expected)
        torch.testing.assert_close(command_target_state(env), target_expected)

        # CRL goal/achieved obs read cmd_buf through the command's payload (the
        # command shell no longer owns cmd_buf); regression for the env-local
        # target/achieved position observations.
        torch.testing.assert_close(target_pos_env(env), target_root[:, :3] - origins)
        torch.testing.assert_close(achieved_pos_env(env), current_root[:, :3] - origins)

    def test_get_task_done_triggers_when_delta_nonpositive(self):
        """``get_task_done`` is true exactly when the hold delta has drained to 0."""
        table = _make_task_table()
        env = _make_env(num_envs=4, device=DEVICE)
        term = _make_command_term(env, table)
        term._payload.cmd_buf[:, 1, term._payload.time_idx] = torch.tensor([0.5, 0.0, -0.1, 1.0], device=DEVICE)
        done = term.get_task_done()
        torch.testing.assert_close(done, torch.tensor([False, True, True, False], device=DEVICE))
        reward = term.get_task_reward()
        torch.testing.assert_close(reward, torch.tensor([0.0, 1.0, 1.0, 0.0], device=DEVICE))

    def test_update_command_writes_per_group_error_norms(self):
        """``_update_command`` writes root norms plus max foot-position error."""
        table = _make_task_table()
        env = _make_env(num_envs=2, device=DEVICE)
        term = _make_command_term(env, table)

        term.cmd_indices[:] = 0
        term._payload.cmd_mask[:, :12] = True
        target = term._payload.cmd_buf[:, 0]
        target.zero_()
        target[0, :3] = torch.tensor([3.0, 4.0, 0.0], device=DEVICE)  # pos norm = 5
        target[0, 3:6] = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)  # rot norm = 1
        target[0, 6:9] = torch.zeros(3, device=DEVICE)  # lin_vel norm = 0
        target[0, 9:12] = torch.tensor([0.0, 2.0, 0.0], device=DEVICE)  # ang_vel = 2
        term._payload.foot_success_mask[0] = True
        term._payload.target_foot_pos_w[0, 0] = torch.tensor([0.03, 0.04, 0.0], device=DEVICE)
        term._payload.target_foot_pos_w[0, 1] = torch.tensor([0.0, 0.0, 0.2], device=DEVICE)

        term._update_command()
        torch.testing.assert_close(
            term._err[0],
            torch.tensor([5.0, 1.0, 0.0, 2.0, 0.2], device=DEVICE),
        )
        torch.testing.assert_close(term._err[1], torch.zeros(5, device=DEVICE))


# ---------------------------------------------------------------------------
# Sampler <-> command-term interplay.
# ---------------------------------------------------------------------------


class _FakeTerminationManager:
    def __init__(self, mask: torch.Tensor):
        self._mask = mask

    def get_term(self, name: str):
        return self._mask


class _FakeCommandManager:
    def __init__(self, term):
        self._term = term

    def get_term(self, name: str):
        return self._term


def _bootstrap_curriculum(env, term, target=0.5, kappa=5.0, history_len=16):
    """Build a curriculum bound to the term and the env, bypassing visual setup."""
    if env.termination_manager is None:
        env.termination_manager = _FakeTerminationManager(
            torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
    cfg = SimpleNamespace(
        params={
            "debug_vis": False,  # skip VisualizationMarkers construction
            **CURRICULUM_BINDS,
            "sampling": SamplerCfg(
                strategies=[
                    BetaSamplingStrategyCfg(target=target, kappa=kappa, weight=1.0, success_rate_bind="success_rates")
                ],
                eps=1e-8,
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=history_len),
        }
    )
    env.command_manager = _FakeCommandManager(term)
    curriculum = success_rate_sampler(cfg=cfg, env=env)
    return curriculum


def _call_curriculum(curriculum, env, env_ids, **kwargs):
    return curriculum(
        env,
        env_ids,
        **CURRICULUM_BINDS,
        sampling=curriculum._sampling_cfg,
        success_monitor_cfg=curriculum.success_monitor.cfg,
        **kwargs,
    )


class TestCurriculum:
    """The curriculum drives cmd_indices and the command term populates cmd_buf/targets."""

    def test_binds_success_rate_as_alias(self):
        """The term's ``success_rates`` must be the *same tensor* as the monitor's rate.

        This is the zero-copy contract: reward functions can read
        ``goal_term.success_rates`` and see updates without an explicit copy.
        """
        table = _make_task_table()
        env = _make_env(num_envs=8, device=DEVICE)
        term = _make_command_term(env, table)
        term.randomize_command_indices = False
        curriculum = _bootstrap_curriculum(env, term)

        assert term.success_rates is not None
        assert term.success_rates.data_ptr() == curriculum.success_monitor.success_rate.data_ptr()

    def test_curriculum_writes_command_owned_indices(self):
        """Curriculum writes command-owned task rows without replacing the tensor."""
        table = _make_task_table()
        env = _make_env(num_envs=8, device=DEVICE)
        term = _make_command_term(env, table)

        # sanity check: before curriculum ownership, _resample_command samples rows.
        term.cmd_indices[:] = 0
        term._resample_command(torch.arange(env.num_envs, device=DEVICE))
        # With num_tasks=20 and 8 draws the chance of all zeros is astronomically small.
        assert not (term.cmd_indices == 0).all()

        term.randomize_command_indices = False
        command_indices_ptr = term.cmd_indices.data_ptr()
        _bootstrap_curriculum(env, term)
        assert term.cmd_indices.data_ptr() == command_indices_ptr
        term.cmd_indices[:] = torch.arange(env.num_envs, device=DEVICE) % table.num_tasks
        term._resample_command(torch.arange(env.num_envs, device=DEVICE))
        expected_indices = torch.arange(env.num_envs, device=DEVICE) % table.num_tasks
        assert torch.equal(term.cmd_indices, expected_indices)
        offsets = table.offsets
        ids = term._payload.cmd_ids.long()
        assert ((term.cmd_indices >= offsets[ids]) & (term.cmd_indices < offsets[ids + 1])).all()

    def test_success_update_uses_cmd_indices_before_overwrite(self):
        """Ordering invariant: monitor sees the *previous* cmd_indices, not the new ones.

        If the order were reversed the monitor would credit the newly-sampled
        task with an outcome that actually belonged to the previous task. We
        verify this by pre-seeding cmd_indices, marking an env as successful,
        and checking the success rate lands on the *pre-call* index.
        """
        torch.manual_seed(7)
        table = _make_task_table(pos_tasks=5, pose_tasks=5)
        env = _make_env(num_envs=4, device=DEVICE)
        term = _make_command_term(env, table)

        success_mask = torch.tensor([True, True, False, False], device=DEVICE)
        env.termination_manager = _FakeTerminationManager(success_mask)

        term.randomize_command_indices = False
        curriculum = _bootstrap_curriculum(env, term, history_len=50)

        # Seed the "previous" indices: env i -> task 2*i so we can read them back
        prev_indices = torch.tensor([0, 2, 4, 6], dtype=torch.long, device=DEVICE)
        term.cmd_indices[:] = prev_indices

        env_ids = torch.arange(env.num_envs, device=DEVICE)
        _call_curriculum(curriculum, env, env_ids)

        # Monitor should have counted exactly one event per prev_index[i].
        for env_i, task_i in enumerate(prev_indices.tolist()):
            expected_rate = 1.0 if success_mask[env_i].item() else 0.0
            assert float(curriculum.success_monitor.success_rate[task_i].item()) == pytest.approx(expected_rate)
            assert int(curriculum.success_monitor.success_size[task_i].item()) == 1

        # cmd_indices was overwritten after the success update ran.
        assert not torch.equal(term.cmd_indices, prev_indices)

    def test_end_to_end_loop_converges_success_rates(self):
        """Simulate multiple episodes: envs with large error never succeed, the
        one env with zero error always succeeds. The monitor should end with
        per-task rates that match that pattern.
        """
        torch.manual_seed(11)
        table = _make_task_table(pos_tasks=3, pose_tasks=3)
        env = _make_env(num_envs=4, device=DEVICE)
        term = _make_command_term(env, table)

        success_mask = torch.tensor([True, False, False, False], device=DEVICE)
        env.termination_manager = _FakeTerminationManager(success_mask)
        term.randomize_command_indices = False
        curriculum = _bootstrap_curriculum(env, term, history_len=8)
        env_ids = torch.arange(env.num_envs, device=DEVICE)

        # Pin each env to a fixed task so we get repeatable monitor updates.
        pinned = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=DEVICE)
        for _ in range(8):
            term.cmd_indices[:] = pinned
            _call_curriculum(curriculum, env, env_ids)

        # Task 0 was always the "successful" env -> rate close to 1.
        # Tasks 1/2/3 were always failures -> rate close to 0.
        assert float(curriculum.success_monitor.success_rate[0].item()) == pytest.approx(1.0)
        assert float(curriculum.success_monitor.success_rate[1].item()) == pytest.approx(0.0)
        assert float(curriculum.success_monitor.success_rate[2].item()) == pytest.approx(0.0)
        assert float(curriculum.success_monitor.success_rate[3].item()) == pytest.approx(0.0)
        # The alias on term sees the same numbers (zero-copy contract).
        torch.testing.assert_close(term.success_rates, curriculum.success_monitor.success_rate)
