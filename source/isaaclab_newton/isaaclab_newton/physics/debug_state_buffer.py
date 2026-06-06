# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug state buffer for NaN incident replay in Newton physics."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from .newton_manager_cfg import ReplayBufferCfg

if TYPE_CHECKING:
    from newton import Model, State

logger = logging.getLogger(__name__)

_MAX_BUFFER_SIZE = 2000

# State attribute -> world-layout "kind" used to slice it per env.
_STATE_KIND = {"body_q": "body", "body_qd": "body", "joint_q": "coord", "joint_qd": "dof"}

# Newton ``Contacts`` fields dumped on the cold (NaN) path, as ``(attr, export_suffix)``.
_CONTACT_DUMP_FIELDS = (
    ("rigid_contact_normal", "normal"),
    ("rigid_contact_point0", "point0"),
    ("rigid_contact_point1", "point1"),
    ("rigid_contact_margin0", "margin0"),
    ("rigid_contact_margin1", "margin1"),
    ("rigid_contact_thickness0", "thickness0"),
    ("rigid_contact_thickness1", "thickness1"),
)

# Newton ``Contacts`` fields cloned in the hot path for the opt-in step-replay buffer.
_REPLAY_CONTACT_FIELDS = (
    "rigid_contact_count", "rigid_contact_shape0", "rigid_contact_shape1",
    "rigid_contact_point0", "rigid_contact_point1", "rigid_contact_normal",
    "rigid_contact_offset0", "rigid_contact_offset1", "rigid_contact_margin0",
    "rigid_contact_margin1", "rigid_contact_thickness0", "rigid_contact_thickness1",
    "rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction",
)  # fmt: skip
# MJWarp ``Data`` vectors cloned in the hot path for the replay buffer.
_REPLAY_SOLVER_FIELDS = ("qacc_warmstart", "qacc_smooth", "qfrc_smooth", "qacc", "qvel", "qfrc_constraint")
# MJWarp ``efc`` rows cloned only when full-context replay is requested.
_REPLAY_EFC_FIELDS = ("force", "D", "aref", "vel", "pos")


def _gather(arr, idx: torch.Tensor | None) -> np.ndarray | None:
    """Move a Warp array to numpy, slicing rows by ``idx`` first when given."""
    if arr is None:
        return None
    return wp.to_torch(arr)[idx].cpu().numpy() if idx is not None else arr.numpy()


def _stack_states(states: list[State], attr: str, idx: torch.Tensor | None) -> np.ndarray | None:
    """Stack one ``State`` attribute across a ring of snapshots, optionally sliced to ``idx``.

    Returns ``None`` if the attribute is absent on the states.
    """
    out: list[np.ndarray] = []
    for state in states:
        arr = getattr(state, attr)
        if arr is None:
            return None
        out.append(_gather(arr, idx))
    return np.stack(out)


class _WorldLayout:
    """Per-world index ranges into the flat body/joint arrays.

    Centralizes the three ``*_world_start`` arrays so NaN localization and per-env slicing share one
    implementation. A *kind* selects the array: ``"body"``, ``"coord"`` (joint coordinates), or
    ``"dof"`` (joint velocities).
    """

    _KIND_ATTR = {
        "body": "body_world_start",
        "coord": "joint_coord_world_start",
        "dof": "joint_dof_world_start",
    }

    def __init__(self, model: Model) -> None:
        self.world_count: int = int(model.world_count) if int(model.world_count) > 1 else 0
        self._starts: dict[str, np.ndarray] = {}
        if self.world_count:
            for kind, attr in self._KIND_ATTR.items():
                arr = getattr(model, attr)
                assert arr is not None
                self._starts[kind] = arr.numpy()

    @property
    def is_multi(self) -> bool:
        """True when the model has more than one world."""
        return self.world_count > 0

    def index(self, kind: str, envs: list[int]) -> torch.Tensor | None:
        """1-D index tensor covering ``envs``' ranges for ``kind``, or ``None`` if empty/unknown."""
        starts = self._starts.get(kind)
        if starts is None:
            return None
        idx: list[int] = []
        for w in envs:
            if w + 1 < starts.shape[0]:
                idx.extend(range(int(starts[w]), int(starts[w + 1])))
        return torch.tensor(idx, dtype=torch.long) if idx else None

    def bad_envs(self, state: State, exclude: set[int]) -> list[int]:
        """World-ids whose state slice contains NaN, skipping any in ``exclude``."""
        tensors = {
            attr: (wp.to_torch(getattr(state, attr)) if getattr(state, attr) is not None else None)
            for attr in _STATE_KIND
        }
        bad: list[int] = []
        for w in range(self.world_count):
            if w in exclude:
                continue
            for attr, kind in _STATE_KIND.items():
                t = tensors[attr]
                starts = self._starts.get(kind)
                if t is None or starts is None:
                    continue
                if torch.isnan(t[int(starts[w]) : int(starts[w + 1])]).any().item():
                    bad.append(w)
                    break
        return bad


class DebugStateBuffer:
    """Rolling buffer of Newton state snapshots with GPU-side NaN detection.

    On every step the current state is copied into a ring buffer (GPU-GPU via ``State.assign``). NaN
    detection runs entirely on GPU using a fused ``torch.isnan`` + ``any`` -- no data leaves the device
    on the hot path. Only when a NaN is found does the buffer dump to CPU for export.

    With multiple worlds (replicated envs) the buffer identifies which env(s) contain NaN, exports only
    those slices, and suppresses future detection of already-exported env_ids (NaN is sticky). After
    :attr:`max_exports` exports, :attr:`nan_halt` is set and subsequent calls are no-ops; the caller
    (Newton manager) should check ``nan_halt`` and raise to stop simulation.

    Modes (all share one ring + export path):

    * default -- per env-step: :meth:`capture_pre` records the pre-physics input, :meth:`step` records
      the post-physics output and checks for NaN.
    * per-substep (``per_substep=True``, CUDA graph must be off) -- :meth:`observe_substep` records and
      checks every solver substep, exporting the last finite substep as the ``pre_*`` state so the NaN's
      birthplace is captured before it propagates.
    * step-replay (``replay_cfg.enabled``) -- :meth:`record_step_replay_pre` / :meth:`record_step_replay_post`
      record pre/post state plus selected control/contact/solver vectors per substep.
    """

    def __init__(
        self,
        model: Model,
        buffer_size: int,
        export_path: str = ".",
        export_envs_only: bool = True,
        max_exports: int = 1,
        scene_exporter: Callable[[str, list[int]], None] | None = None,
        collision_pipeline=None,
        per_substep: bool = False,
        replay_cfg: ReplayBufferCfg = ReplayBufferCfg(),
    ) -> None:
        """Initialize the debug state buffer.

        Args:
            model: Finalized Newton model (used to allocate state clones and read world layout).
            buffer_size: Number of state snapshots to keep. Capped at :data:`_MAX_BUFFER_SIZE`.
            export_path: Directory for npz export.
            export_envs_only: When True and the model has multiple worlds, export only the env(s) that
                contain NaN.
            max_exports: Maximum number of NaN export events before halting. Each event exports a
                distinct set of newly-NaN env_ids.
            scene_exporter: Optional callable ``(usd_path, env_ids) -> None`` that exports USD prim
                subtrees for the given env_ids. Called once per export event; an empty ``env_ids``
                means export the whole scene (single-env case).
            collision_pipeline: Live collision pipeline (used on NaN to re-collide the pre-state).
            per_substep: Enable per-substep deep capture (requires the CUDA graph to be off).
            replay_cfg: Configuration for the opt-in per-substep replay buffer.
        """
        size = min(max(int(buffer_size), 1), _MAX_BUFFER_SIZE)
        self._ring: list[State] = [model.state() for _ in range(size)]
        self._size: int = size
        self._write_idx: int = 0
        self._layout = _WorldLayout(model)

        # Pre-physics snapshot of the in-flight step plus that step's per-env episode length. Lets the
        # export distinguish a NaN already present in the step *input* (e.g. from a reset that fired at
        # the end of the previous env-step) from one produced by the physics step itself.
        self._pending_pre: State = model.state()
        self._last_episode_length = None

        # Deep per-substep mode: track the last finite substep so it can be exported as the "pre".
        self._per_substep: bool = bool(per_substep)
        self._substep_pre: State = model.state()
        self._substep_counter: int = 0
        self._last_finite_substep: int = -1
        self._substep_meta: tuple[int, int, int] | None = None

        # Opt-in step-replay buffer (records pre/post transitions for offline replay).
        self._replay_cfg = replay_cfg
        self._replay_enabled: bool = replay_cfg.enabled
        if replay_cfg.record_mjwarp_context:
            from mujoco_warp._src import solver as mjw_solver

            mjw_solver._SOLVE_SNAPSHOT = True  # noqa: SLF001
        self._replay_pre: list[State] = [model.state() for _ in range(size)] if self._replay_enabled else []
        self._replay_post: list[State] = [model.state() for _ in range(size)] if self._replay_enabled else []
        self._replay_pre_aux: list[dict | None] = [None] * size
        self._replay_post_aux: list[dict | None] = [None] * size
        self._replay_meta: list[dict] = [{} for _ in range(size)]
        self._replay_write_idx: int = 0
        self._replay_current_idx: int | None = None

        self._export_path: str = export_path
        self._export_envs_only: bool = export_envs_only
        self._max_exports: int = max(int(max_exports), 1)
        self._scene_exporter = scene_exporter
        self._export_count: int = 0
        self._nan_halt: bool = False
        self._exported_envs: set[int] = set()

        # Optional collision pipeline + scratch contacts. On NaN we run a single collide() on the stored
        # pre-physics state to capture the contact set that fed the failing step (cold path only).
        self._collision_pipeline = collision_pipeline
        self._pre_contacts = collision_pipeline.contacts() if collision_pipeline is not None else None

        assert model.shape_world is not None
        assert model.body_flags is not None
        assert model.joint_armature is not None
        self._shape_world: np.ndarray = model.shape_world.numpy()
        self._body_label: list[str] = [str(x) for x in model.body_label]
        self._shape_label: list[str] = [str(x) for x in model.shape_label]
        self._body_flags: np.ndarray = model.body_flags.numpy()
        self._joint_armature: np.ndarray = model.joint_armature.numpy()
        # Live (post-step) contacts and active solver, set each step; dumped on NaN to explain *why*.
        self._last_contacts = None
        self._last_solver = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of state snapshots in the buffer."""
        return self._size

    @property
    def nan_halt(self) -> bool:
        """True after :attr:`max_exports` NaN exports have occurred."""
        return self._nan_halt

    @property
    def per_substep(self) -> bool:
        """True when running in deep per-substep capture mode (CUDA graph must be off)."""
        return self._per_substep

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def capture_pre(self, pre_state: State) -> None:
        """Snapshot the pre-physics-step state of the in-flight env-step.

        Call at the top of the physics step (after the previous env-step's reset/action application,
        before the solver). Paired with :meth:`step` so the export can show the failing step's *input*.

        Args:
            pre_state: Newton state before this step's physics (GPU).
        """
        if self._nan_halt or not self._ring:
            return
        self._pending_pre.assign(pre_state)

    def step(
        self,
        current_state: State,
        sim_time: float,
        episode_length=None,
        contacts=None,
        solver=None,
        collision_pipeline=None,
    ) -> None:
        """Copy state into the ring, check for NaN, and export if found.

        Everything except the final export stays on GPU. No-op when :attr:`nan_halt` is True.

        Args:
            current_state: Current Newton state (GPU), post-physics.
            sim_time: Simulation time [s] at this step.
            episode_length: Per-env episode length, either a tensor or a no-arg callable returning the
                live tensor (callable is preferred -- the RL env may reallocate the buffer).
            contacts: Live Newton ``Contacts`` (post-step) -- dumped on NaN.
            solver: Active solver (e.g. ``SolverMuJoCo``); its ``mjw_data`` internals are dumped on NaN.
            collision_pipeline: Live collision pipeline for the pre-state re-collide (``None`` at
                construction, so it is refreshed here).
        """
        if self._nan_halt or not self._ring:
            return
        self._record_step_context(episode_length, contacts, solver, collision_pipeline)
        self._ring[self._write_idx].assign(current_state)
        self._write_idx = (self._write_idx + 1) % self._size
        nan_detected, bad_envs = self._detect_nan(current_state)
        if nan_detected:
            self._export(sim_time, bad_envs)

    def observe_substep(
        self,
        current_state: State,
        sim_time: float,
        episode_length=None,
        contacts=None,
        solver=None,
        collision_pipeline=None,
        substep_idx: int = 0,
    ) -> None:
        """Deep-mode per-substep NaN check (requires the CUDA graph to be OFF).

        Records each solver substep into the ring and, on the first substep whose state contains NaN,
        exports the **last finite substep** as the ``pre_*`` state plus the solver internals --
        pinpointing where the NaN is born before it propagates.

        Args:
            current_state: Post-substep Newton state (GPU).
            sim_time: Simulation time [s].
            episode_length: Per-env episode-length tensor or no-arg callable.
            contacts: Live Newton contacts at this substep.
            solver: Active solver (``mjw_data`` dumped on NaN).
            collision_pipeline: Live collision pipeline.
            substep_idx: Index of this substep within the current physics dispatch.
        """
        if self._nan_halt or not self._ring:
            return
        self._record_step_context(episode_length, contacts, solver, collision_pipeline)
        self._substep_counter += 1
        self._ring[self._write_idx].assign(current_state)
        self._write_idx = (self._write_idx + 1) % self._size
        nan_detected, bad_envs = self._detect_nan(current_state)
        if nan_detected:
            # The last finite substep is the input to this failing substep -> export it as pre_*.
            self._pending_pre.assign(self._substep_pre)
            self._substep_meta = (int(substep_idx), int(self._last_finite_substep), int(self._substep_counter))
            self._export(sim_time, bad_envs)
        else:
            self._substep_pre.assign(current_state)
            self._last_finite_substep = int(substep_idx)

    def record_step_replay_pre(
        self,
        pre_state: State,
        control=None,
        contacts=None,
        solver=None,
        sim_time: float = 0.0,
        substep_idx: int = 0,
        episode_length=None,
    ) -> None:
        """Record replay input for one solver substep (opt-in via :class:`ReplayBufferCfg`)."""
        if not self._replay_enabled or self._nan_halt:
            return
        idx = self._replay_write_idx
        self._replay_current_idx = idx
        if self._replay_cfg.record_state:
            self._replay_pre[idx].assign(pre_state)
        self._replay_pre_aux[idx] = self._capture_replay_aux(control=control, contacts=contacts, solver=solver)
        self._replay_post_aux[idx] = None
        self._replay_meta[idx] = {"sim_time": float(sim_time), "substep_idx": int(substep_idx)}

    def record_step_replay_post(self, post_state: State, solver=None) -> None:
        """Record replay output for the latest solver substep."""
        if not self._replay_enabled or self._nan_halt or self._replay_current_idx is None:
            return
        idx = self._replay_current_idx
        if self._replay_cfg.record_state:
            self._replay_post[idx].assign(post_state)
        self._replay_post_aux[idx] = self._capture_replay_aux(solver=solver)
        self._replay_write_idx = (idx + 1) % self._size
        self._replay_current_idx = None

    def _record_step_context(self, episode_length, contacts, solver, collision_pipeline) -> None:
        """Stash the per-step references the cold-path export needs."""
        self._last_episode_length = episode_length() if callable(episode_length) else episode_length
        self._last_contacts = contacts
        self._last_solver = solver
        if collision_pipeline is not None:
            self._collision_pipeline = collision_pipeline
            if self._pre_contacts is None:
                self._pre_contacts = collision_pipeline.contacts()

    @staticmethod
    def _clone_wp_array(arr):
        out = wp.empty_like(arr)
        wp.copy(out, arr)
        return out

    def _capture_replay_aux(self, control=None, contacts=None, solver=None) -> dict:
        """Clone the configured control/contact/solver vectors for one replay record (kept on GPU)."""
        aux: dict = {}
        cfg = self._replay_cfg
        if cfg.record_control and control is not None:
            aux["control_joint_f"] = self._clone_wp_array(control.joint_f)
        if cfg.record_contacts and contacts is not None:
            for field in _REPLAY_CONTACT_FIELDS:
                aux[f"contact_{field}"] = self._clone_wp_array(getattr(contacts, field))
        if cfg.record_solver and solver is not None:
            mjw = solver.mjw_data
            for field in _REPLAY_SOLVER_FIELDS:
                aux[f"mjw_{field}"] = self._clone_wp_array(getattr(mjw, field))
            if cfg.record_mjwarp_context:
                for field in _REPLAY_EFC_FIELDS:
                    aux[f"mjw_efc_{field}"] = self._clone_wp_array(getattr(mjw.efc, field))
        return aux

    def _detect_nan(self, state: State) -> tuple[bool, list[int]]:
        """GPU-side NaN check. Returns ``(has_nan, bad_env_ids)`` (env ids empty if single-world)."""
        arrays = [
            wp.to_torch(getattr(state, attr)).flatten() for attr in _STATE_KIND if getattr(state, attr) is not None
        ]
        if not arrays:
            return False, []
        if not torch.isnan(torch.cat(arrays)).any().item():
            return False, []
        if not self._layout.is_multi:
            return True, []
        bad = self._layout.bad_envs(state, self._exported_envs)
        return len(bad) > 0, bad

    # ------------------------------------------------------------------
    # Export (cold path -- only on NaN)
    # ------------------------------------------------------------------

    def _export(self, sim_time: float, bad_envs: list[int]) -> None:
        """Dump the ring buffer (and diagnostics) to npz. Only called when a NaN is detected."""
        n = self._size
        ordered = [self._ring[(self._write_idx + i) % n] for i in range(n)]
        sliced_envs = bad_envs if (self._export_envs_only and bad_envs) else []

        data = self._dump_states(ordered, sliced_envs)
        if sliced_envs:
            data["exported_env_ids"] = np.array(bad_envs, dtype=np.int32)
        data["buffer_size"] = n
        data["sim_time"] = sim_time
        if self._substep_meta is not None:
            data["per_substep_capture"] = np.int64(1)
            data["nan_substep_idx"] = np.int64(self._substep_meta[0])
            data["last_finite_substep_idx"] = np.int64(self._substep_meta[1])
            data["substep_counter"] = np.int64(self._substep_meta[2])

        self._add_pre_and_episode_length(data, bad_envs)
        self._add_contacts_and_solver(data, bad_envs)
        self._add_step_replay(data, bad_envs)

        os.makedirs(self._export_path, exist_ok=True)
        stem = f"nan_replay_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        npz_path = os.path.join(self._export_path, f"{stem}.npz")
        np.savez_compressed(npz_path, **data)

        if self._scene_exporter is not None:
            usd_path = os.path.join(self._export_path, f"{stem}.usd")
            try:
                self._scene_exporter(usd_path, bad_envs)
                logger.info("Exported scene for envs %s to %s", bad_envs or "all", usd_path)
            except Exception:
                logger.exception("Failed to export scene USD to %s", usd_path)

        logger.error("NaN detected (envs %s). Exported %d snapshots to %s", bad_envs or "all", n, npz_path)

        self._exported_envs.update(bad_envs)
        self._export_count += 1
        if self._export_count >= self._max_exports:
            self._nan_halt = True
            logger.error("Reached max NaN exports (%d). Halting debug state buffer.", self._max_exports)

    def _dump_states(self, ordered: list[State], envs: list[int]) -> dict:
        """Stack each state attribute across the ring, sliced to ``envs`` when nonempty."""
        data: dict = {}
        for attr, kind in _STATE_KIND.items():
            idx = self._layout.index(kind, envs) if envs else None
            stacked = _stack_states(ordered, attr, idx)
            if stacked is not None:
                data[attr] = stacked
        return data

    def _add_pre_and_episode_length(self, data: dict, bad_envs: list[int]) -> None:
        """Add the pre-physics snapshot (``pre_*``), per-env episode length, and labels."""
        envs = bad_envs if (self._export_envs_only and bad_envs and self._layout.is_multi) else []
        pre = self._pending_pre
        for attr, kind in _STATE_KIND.items():
            arr = _gather(getattr(pre, attr), self._layout.index(kind, envs) if envs else None)
            if arr is not None:
                data[f"pre_{attr}"] = arr

        el = self._last_episode_length
        if el is not None:
            el_t = el if isinstance(el, torch.Tensor) else wp.to_torch(el)
            el_np = el_t.detach().cpu().numpy().reshape(-1)
            data["episode_length"] = el_np[bad_envs] if (self._export_envs_only and bad_envs) else el_np

        body_idx = self._layout.index("body", envs) if envs else None
        data["body_label"] = np.array(
            [self._body_label[i] for i in body_idx.tolist()] if body_idx is not None else self._body_label
        )
        data["shape_label"] = np.array(self._shape_label)

    def _add_contacts_and_solver(self, data: dict, bad_envs: list[int]) -> None:
        """Dump the pre-state re-collide contacts, the live contacts, and the solver internals."""
        if self._collision_pipeline is not None and self._pre_contacts is not None:
            try:
                self._collision_pipeline.collide(self._pending_pre, self._pre_contacts)
                self._dump_contacts_obj(data, self._pre_contacts, "pre_contact", bad_envs)
                data["shape_world"] = self._shape_world
            except Exception:
                logger.exception("Failed to re-collide pre-state in NaN export")
        try:
            self._dump_contacts_obj(data, self._last_contacts, "contact", bad_envs)
        except Exception:
            logger.exception("Failed to dump live contacts in NaN export")
        # Best-effort: solver internals read the mjwarp solve snapshot, which only exists when mjwarp
        # is patched (FACTORY_SOLVE_SNAPSHOT) -- never let its absence cost the whole capture.
        if self._last_solver is not None:
            try:
                self._add_solver_internals(data, bad_envs)
            except Exception:
                logger.exception("Failed to dump solver internals in NaN export")

    def _dump_contacts_obj(self, data: dict, c, prefix: str, bad_envs: list[int]) -> None:
        """Dump a populated Newton ``Contacts`` filtered to the NaN'd env(s) via the shape->world map."""
        if c is None:
            return
        cnt = int(wp.to_torch(c.rigid_contact_count).reshape(-1)[0].item())
        data[f"{prefix}_count_total"] = np.int64(cnt)
        cnt = max(0, min(cnt, int(c.rigid_contact_shape0.shape[0])))
        sh0 = wp.to_torch(c.rigid_contact_shape0)[:cnt].cpu().numpy()
        sh1 = wp.to_torch(c.rigid_contact_shape1)[:cnt].cpu().numpy()
        if self._export_envs_only and bad_envs:
            sw = self._shape_world
            n = len(sw)
            w0 = np.where((sh0 >= 0) & (sh0 < n), sw[np.clip(sh0, 0, n - 1)], -2)
            w1 = np.where((sh1 >= 0) & (sh1 < n), sw[np.clip(sh1, 0, n - 1)], -2)
            sel = np.where(np.isin(w0, bad_envs) | np.isin(w1, bad_envs))[0]
        else:
            sel = np.arange(cnt)
        data[f"{prefix}_count_env"] = np.int64(len(sel))
        data[f"{prefix}_shape0"] = sh0[sel]
        data[f"{prefix}_shape1"] = sh1[sel]
        for field, suffix in _CONTACT_DUMP_FIELDS:
            arr = getattr(c, field, None)  # Newton's Contacts schema varies by version; skip absent fields
            if arr is None:
                continue
            data[f"{prefix}_{suffix}"] = wp.to_torch(arr)[:cnt].cpu().numpy()[sel]

    def _add_solver_internals(self, data: dict, bad_envs: list[int]) -> None:
        """Dump the MJWarp solver's internal state for the NaN'd world -- the data that explains *why*
        the solve produced NaN, raw (derive forces/penetration downstream):

        * convergence (``solver_niter``) and the mass-matrix factorization (``qLDiagInv``: 1/pivot of
          the LDL factor of M -- a huge entry means a ~zero pivot / singular M);
        * per-world forces/accel (``qfrc_constraint``, ``qfrc_smooth``, ``qacc``, ``qvel``);
        * constraint rows (``efc.*``): ``force``, ``aref``, ``D`` and the sparse Jacobian (so the active
          block's rank can be checked downstream);
        * contacts (``contact.*``) filtered by ``worldid`` plus the geom->Newton-shape map;
        * the GPU-side solve snapshot (the latched failing-iteration Hessian and iterate);
        * ``mjw_nan_summary_*``: per-array NaN counts to localize the failure at a glance.
        """
        solver = self._last_solver
        mjw = solver.mjw_data
        world = int(bad_envs[0]) if bad_envs else None
        nworld = int(mjw.nworld)
        summary_keys: list[str] = []
        summary_counts: list[int] = []

        def _record(key, arr):
            if np.issubdtype(arr.dtype, np.floating):
                summary_keys.append(key)
                summary_counts.append(int(np.isnan(arr).sum()))

        def _dump_world(container, names, prefix):
            for name in names:
                arr = getattr(container, name).numpy()
                key = f"{prefix}_{name}"
                _record(key, arr)
                if world is not None and arr.ndim >= 1 and arr.shape[0] == nworld:
                    data[key] = arr[world]
                elif arr.size <= 100_000:
                    data[key] = arr

        # (1) per-world scalars/vectors and the mass-matrix factorization.
        _dump_world(mjw, [
            "solver_niter", "ne", "nf", "nl", "nefc",
            "qacc", "qvel", "qfrc_constraint", "qfrc_smooth", "qfrc_passive",
            "qM", "qLDiagInv", "qfrc_actuator", "act",
        ], "mjw")  # fmt: skip
        # (2) constraint rows (efc): forces plus the sparse Jacobian structure for a rank check.
        _dump_world(
            mjw.efc,
            ["force", "aref", "pos", "margin", "D", "vel", "Jqvel", "type",
             "J", "J_rownnz", "J_rowadr", "J_colind"],
            "mjw_efc",
        )  # fmt: skip
        # (3) contacts: flat (naconmax, ...) filtered by worldid == world.
        contact = mjw.contact
        if world is not None:
            nacon = int(mjw.nacon.numpy().reshape(-1)[0])
            csel = np.where(contact.worldid.numpy()[:nacon] == world)[0]
            data["mjw_contact_count_env"] = np.int64(len(csel))
            for name in ("dist", "pos", "frame", "geom", "efc_address",
                         "includemargin", "friction", "solref", "solimp", "dim"):  # fmt: skip
                arr = getattr(contact, name).numpy()
                _record(f"mjw_contact_{name}", arr)
                if arr.ndim >= 1 and len(csel):
                    data[f"mjw_contact_{name}"] = arr[csel]
            # Map each contact's MJWarp geom ids -> Newton shape indices (shape-identifiable downstream).
            if len(csel):
                row = solver.mjc_geom_to_newton_shape.numpy()
                row = row[world] if row.ndim == 2 else row
                g = data["mjw_contact_geom"]
                ns = np.where((g >= 0) & (g < len(row)), row[np.clip(g, 0, len(row) - 1)], -1)
                data["mjw_contact_newton_shape0"] = ns[:, 0]
                data["mjw_contact_newton_shape1"] = ns[:, 1]

        # (4) model-level constants: equality definitions, masses, armature, DOF/body ids.
        mjm = solver.mjw_model
        for name in ("eq_type", "eq_obj1id", "eq_obj2id", "eq_objtype",
                     "body_mass", "body_inertia", "dof_bodyid", "dof_jntid", "jnt_bodyid",
                     "dof_armature", "body_weldid", "body_invweight0", "dof_invweight0"):  # fmt: skip
            arr = getattr(mjm, name).numpy()
            if arr.ndim >= 1 and arr.shape[0] == nworld:
                arr = arr[world] if world is not None else arr[0]
            if arr.size <= 20000:
                data[f"mjw_model_{name}"] = arr

        # (5) env-0 body flags / armature and the DOF mapping arrays (kinematic-armature localization).
        body0 = self._layout.index("body", [0])
        if body0 is not None:
            data["body_flags_env0"] = self._body_flags[int(body0.min()) : int(body0.max()) + 1]
        data["joint_armature_env0"] = self._joint_armature[:64]
        for attr, key in (
            ("mjc_dof_to_newton_dof", "mjw_mjc_dof_to_newton_dof"),
            ("newton_dof_to_body", "mjw_newton_dof_to_body"),
            ("mjc_dof_to_newton_body", "mjw_mjc_dof_to_newton_body"),
        ):
            arr = getattr(solver, attr).numpy()
            if arr.ndim == 2 and world is not None and world < arr.shape[0]:
                arr = arr[world]
            if arr.size <= 20000:
                data[key] = arr
        nmdl = solver.model
        for name in ("joint_child", "joint_parent", "joint_qd_start", "joint_dof_dim", "joint_type"):
            data[f"nmodel_{name}"] = getattr(nmdl, name).numpy()[:48]

        # (6) GPU-side MJWarp solve snapshot (detection + copy happen in-kernel; export serializes it).
        from mujoco_warp._src import solver as mjw_solver

        dbg = mjw_solver._SOLVER_DEBUG_LAST  # noqa: SLF001
        if dbg is not None:
            for name in (
                "debug_meta", "debug_qacc_warmstart", "debug_qacc_smooth", "debug_qacc", "debug_jaref",
                "debug_search", "debug_mv", "debug_jv", "debug_grad", "debug_mgrad", "debug_force",
                "debug_h", "debug_efc_state", "debug_efc_D",
            ):  # fmt: skip
                if hasattr(dbg, name):
                    data[f"mjw_solve_{name}"] = getattr(dbg, name).numpy()

        if summary_keys:
            data["mjw_nan_summary_keys"] = np.array(summary_keys)
            data["mjw_nan_summary_counts"] = np.array(summary_counts, dtype=np.int64)
        if world is not None:
            data["mjw_world"] = np.int64(world)
        data["mjw_nworld"] = np.int64(nworld)

    def _add_step_replay(self, data: dict, bad_envs: list[int]) -> None:
        """Add the opt-in per-substep replay records (state + cloned aux) to the export."""
        if not self._replay_enabled or not self._replay_pre:
            return
        n = self._size
        order = [(self._replay_write_idx + i) % n for i in range(n)]
        envs = bad_envs if (self._replay_cfg.export_envs_only and bad_envs and self._layout.is_multi) else []

        if self._replay_cfg.record_state:
            ordered_pre = [self._replay_pre[i] for i in order]
            ordered_post = [self._replay_post[i] for i in order]
            for attr, kind in _STATE_KIND.items():
                idx = self._layout.index(kind, envs) if envs else None
                pre = _stack_states(ordered_pre, attr, idx)
                post = _stack_states(ordered_post, attr, idx)
                if pre is not None:
                    data[f"replay_pre_{attr}"] = pre
                if post is not None:
                    data[f"replay_post_{attr}"] = post

        data["replay_meta_sim_time"] = np.array(
            [self._replay_meta[i].get("sim_time", np.nan) for i in order], dtype=np.float64
        )
        data["replay_meta_substep_idx"] = np.array(
            [self._replay_meta[i].get("substep_idx", -1) for i in order], dtype=np.int32
        )
        data["replay_buffer_size"] = np.int64(n)

        aux_keys = sorted(
            {key for aux in self._replay_pre_aux + self._replay_post_aux if aux is not None for key in aux}
        )
        for key in aux_keys:
            pre_vals = [self._replay_pre_aux[i].get(key) if self._replay_pre_aux[i] else None for i in order]
            post_vals = [self._replay_post_aux[i].get(key) if self._replay_post_aux[i] else None for i in order]
            if all(v is not None for v in pre_vals):
                data[f"replay_pre_{key}"] = np.stack([v.numpy() for v in pre_vals])
            if all(v is not None for v in post_vals):
                data[f"replay_post_{key}"] = np.stack([v.numpy() for v in post_vals])

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Release the ring buffer."""
        self._ring.clear()
        self._write_idx = 0
        self._size = 0
