Debugging Physics Incidents
===========================

Newton can retain a strict rolling physics history and write a focused artifact
when a state first becomes non-finite. The recorder discovers provider schemas
at physics initialization, records only failed worlds by default, and preserves
enough provenance to validate or replay declared capabilities later.

Configure incident capture
--------------------------

Attach :class:`~isaaclab_newton.physics.NewtonDebugCaptureCfg` to
:attr:`~isaaclab_newton.physics.NewtonCfg.debug_capture`:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import (
        NewtonCfg,
        NewtonDebugCaptureCfg,
        NewtonDebugReplayCfg,
    )

    physics_cfg = NewtonCfg(
        use_cuda_graph=False,
        solver_reset=NewtonCfg.SolverResetCfg(enabled=True),
        debug_capture=NewtonDebugCaptureCfg(
            history_length=200,
            output_dir="./physics_debug",
            record_scene=False,
            failed_worlds_only=True,
            max_incidents=5,
            halt_on_incident=True,
            fail_on_capture_error=True,
            max_gpu_bytes=2 * 1024**3,
            record_model=True,
            record_control=True,
            record_contacts=False,
            record_collision_pipeline=False,
            record_solver=True,
            record_operations=False,
            detect_nonfinite_in=("state", "solver"),
            include_private_fields=True,
            replay=NewtonDebugReplayCfg(
                enabled=True,
                record_state=True,
                record_control=True,
                record_solver=True,
                record_contacts=False,
                record_collision_pipeline=False,
                record_operations=False,
            ),
        ),
    )
    sim_cfg = SimulationCfg(physics=physics_cfg)

Set ``use_cuda_graph=False`` when ``capture_per_substep=True``, replay is
enabled, or incident ``record_operations=True``. Basic per-step incident
capture without operation recording remains compatible with CUDA graphs.
The single ``max_gpu_bytes`` limit covers the state history, pre-state slots,
and replay pre/post histories. Initialization reports the bytes for every field
instead of truncating the recording.

``record_scene=False`` is the default. Set it to ``True`` only when an
optional flattened full-stage USD is useful as static geometry and workflow
context. The exporter validates clone-plan coverage, but does not prune to the
failed worlds or stamp post-step Newton transforms. Incident arrays are the
authoritative dynamic state; missing stage or clone-plan data makes an enabled
scene export a capture error.

``record_control=True`` records the live applied Newton control in every cold
incident even when replay is disabled. This preserves a bad action or force
that produced the failure. Replay ``record_control`` independently retains
control at every transition or substep.

``record_contacts=False`` is solver-neutral and safe by default. Enable it for
a finalized external Newton ``CollisionPipeline`` whose contacts are live at
the capture cadence. Internal MJWarp ``Contacts`` are sensor-reporting mirrors
and are rejected as incident or replay evidence; use ``record_solver`` for
``solver.mjw_data.contact`` and operation capture for the built-in final
``CollisionContext``.

Set ``record_collision_pipeline=True`` only for a solver using Newton's
finalized external ``CollisionPipeline``. That pipeline becomes a required
load-time provider. A solver with internal collision detection should keep this
option disabled and capture its collision internals through ``record_solver``.
Set ``replay.record_collision_pipeline=True`` when the external pipeline must
be retained in every replay pre/post snapshot rather than only the cold incident
snapshot. It requires replay and the finalized pipeline, and its full pre/post
history counts against ``max_gpu_bytes``.

``detect_nonfinite_in`` defaults to ``("state",)``. Add a recorded retained
provider to let its arrays trigger an incident; selecting ``"solver"`` catches
retained Hessian, factorization, and constraint arrays. Supported names are
``"state"``, ``"model"``, ``"control"``, ``"contacts"``, ``"solver"``,
``"collision_pipeline"``, ``"context"``, and ``"operations"``. Recorded
non-state providers require their corresponding recording option. Selecting
``"context"`` scans registered workflow values and requires at least one
capturable context provider when the recorder binds. Transient locals are
visible only through ``"operations"``. Set incident
``record_operations=True`` to retain the latest snapshot and enable detection;
set replay ``record_operations=True`` to retain every transition snapshot.
Either mode requires a compatible operation provider.

:attr:`~isaaclab_newton.physics.NewtonDebugCaptureCfg.detect_nonfinite_include_fields`
and :attr:`~isaaclab_newton.physics.NewtonDebugCaptureCfg.detect_nonfinite_exclude_fields`
limit only the trigger scan; incident archives still contain every configured
field. Patterns are full-path globs and every pattern must match during strict
schema binding, so an upstream rename fails at initialization. Use these filters
to exclude known inactive-capacity NaNs or to scan only a costly diagnostic
workspace without reducing the captured evidence.

Register custom incident triggers
---------------------------------

Use :meth:`~isaaclab_newton.physics.NewtonManager.set_debug_incident_trigger`
for values that remain finite but are physically invalid. Register each trigger
after the simulation has installed ``NewtonCfg.debug_capture`` and before the
solver is initialized. Names must be unique lower-snake-case strings.

The following trigger watches one known joint degree of freedom per world:

.. code-block:: python

    from collections.abc import Mapping

    from isaaclab_newton.physics import NewtonManager


    def make_joint_velocity_spike_trigger(
        joint_row_by_world: Mapping[int, int], velocity_limit: float
    ):
        def joint_velocity_spike(context: NewtonManager.DebugTriggerContext):
            joint_qd = context.state.joint_qd.numpy()
            failed_worlds = tuple(
                world_id
                for world_id, joint_row in sorted(joint_row_by_world.items())
                if abs(joint_qd[joint_row]) > velocity_limit
            )
            if not failed_worlds:
                return None
            return NewtonManager.DebugTriggerResult(
                reason=(
                    "monitored joint velocity exceeded "
                    f"{velocity_limit} m/s or rad/s, depending on joint type"
                ),
                world_ids=failed_worlds,
            )

        return joint_velocity_spike


    NewtonManager.set_debug_incident_trigger(
        "joint_velocity_spike",
        make_joint_velocity_spike_trigger(
            joint_row_by_world={0: 4, 1: 11}, velocity_limit=30.0
        ),
    )

Collision storage differs between solvers, so keep the extraction policy beside
the solver integration instead of assuming a field name. This adapter accepts a
function that returns the maximum impulse [N·s] for each world from
``context.contacts`` or ``context.collision_pipeline``:

.. code-block:: python

    from collections.abc import Callable, Mapping

    from isaaclab_newton.physics import NewtonManager


    def make_collision_impulse_trigger(
        read_impulse_by_world: Callable[
            [NewtonManager.DebugTriggerContext], Mapping[int, float]
        ],
        impulse_limit: float,
    ):
        def collision_impulse(context: NewtonManager.DebugTriggerContext):
            impulse_by_world = read_impulse_by_world(context)
            failed_worlds = tuple(
                world_id
                for world_id, impulse in sorted(impulse_by_world.items())
                if impulse > impulse_limit
            )
            if not failed_worlds:
                return None
            return NewtonManager.DebugTriggerResult(
                reason=f"collision impulse exceeded {impulse_limit} N·s",
                world_ids=failed_worlds,
            )

        return collision_impulse


    NewtonManager.set_debug_incident_trigger(
        "collision_impulse",
        make_collision_impulse_trigger(read_solver_impulses, impulse_limit=100.0),
    )

The ``read_solver_impulses`` function is application code: it should fail if
the solver's expected collision schema is unavailable. For a world-independent
condition, return ``DebugTriggerResult(reason=..., global_scope=True)``. A
result must select at least one unique, in-range world or global scope. Callback
exceptions, invalid results, and late or duplicate registration fail loudly.
Triggers run in deterministic name order and re-arm a scope only after that
trigger returns ``None`` for the scope.

Strict initialization and schema binding
----------------------------------------

The recorder is created after Newton has initialized its state, control,
solver, and contacts. Every provider requested by the configuration is required
at that point. Missing providers, empty required provider schemas, unmatched
include, exclude, or detection patterns, and an insufficient memory budget fail
physics initialization.

Discovery reads dataclass fields, mappings, instance ``vars``, and declared
slots. ``include_private_fields=True`` keeps important solver and model state.
Properties are inventoried without invoking their getters. Unsupported, opaque,
and non-data runtime resources remain explicit inventory entries and produce an
initialization warning; they are not falsely presented as captured data. A
configured pattern that expected one of those paths still fails strictly.

The bound schema is frozen: adding, removing, or
replacing a field, or changing its dtype or shape, is an error with the exact
field path. ``include_fields`` and ``exclude_fields`` accept full-path glob
patterns; every configured pattern must match, so an upstream rename fails
during initialization instead of silently reducing the recording.

Partial artifacts are reserved for failures that occur after initialization,
such as a cold-path scene export error. The state history is written atomically
with ``status=partial`` and a manifest error. With the default
``fail_on_capture_error=True``, the recorder raises after preserving that
artifact.

Artifacts and incident scopes
-----------------------------

Artifacts use the name ``physics_incident_<timestamp>_<event>_<scope>.npz``.
Each archive contains a versioned manifest, array dtype and shape inventory,
SHA-256 checksums, dependency and runtime provenance, provider schemas, errors,
and declared replay capabilities. Payload keys are namespaced, for example
``history__state__joint_q``, ``pre__state__joint_q``,
``incident__control__...``, ``incident__solver__...``, and
``replay__pre__state__...``. Only valid history frames are exported, in chronological order.

With ``failed_worlds_only=True``, a newly failed world receives its own compact
artifact and ``failed_world_ids`` identifies it. A simultaneous failure in
world-independent data receives a separate global artifact with
``incident__is_global`` set. ``max_incidents`` counts incident events rather
than the number of world and global files created by one event.

Halting and environment resets are independent policies. The safe default,
``halt_on_incident=True``, stops after the first new incident. Set it to
``False`` only when continued diagnostics are intentional; a clean observation
re-arms that scope for a later incident, while ``max_incidents`` limits only
artifact retention.
:attr:`~isaaclab_newton.physics.NewtonCfg.solver_reset` clears solver-owned
state after task-authored environment writes when explicitly enabled, regardless
of the incident halt policy.

Capture timing and reset independence
-------------------------------------

Regular capture freezes one transition in this order:

#. Apply model notifications and the explicitly enabled
   :attr:`~isaaclab_newton.physics.NewtonCfg.solver_reset` for task-authored
   environment writes.
#. Refresh forward kinematics, clear the consumed reset masks, and save the
   post-reset, pre-dispatch state before actuator execution.
#. Apply actuators and run physics integration. Replay pre-state, when
   configured, is captured after the actuator writes and immediately before
   each solver transition.
#. Evaluate custom triggers and non-finite providers at ``dispatch_post``, then
   export any incident with the live applied control before the physics step
   returns.

This post-step observation and cold artifact export are synchronous. A task
reset cannot overwrite the failing post-state between integration and export.
The next task-authored reset remains independent: it dirties its normal reset
mask, and the enabled ``solver_reset`` consumes that mask before the next
pre-state even when ``halt_on_incident=False``.

With ``capture_per_substep=True``, each solver substep instead records
post-actuator replay pre-state, integrates once, records replay post-state, and
evaluates at ``solver_post`` with the exact ``substep_idx`` and post-substep ``sim_time``.
An incident with the halt policy enabled stops before another substep can
overwrite it. Trigger callbacks can inspect ``context.phase`` when the same
callback supports either cadence.
When ``halt_on_incident=False`` in a multi-world run, ``pre__*`` remains the
last globally finite substep after any world becomes non-finite. Chronological
``history__*`` frames retain the intervening evidence if another world fails
later.

Register workflow context
-------------------------

Workflow state can be added without coupling it to the Newton recorder. Register
a lazy, lower-snake-case provider before the first simulator reset:

.. code-block:: python

    from isaaclab.envs import ManagerBasedRLEnv


    class MyEnv(ManagerBasedRLEnv):
        def _register_physics_debug_context(self) -> None:
            super()._register_physics_debug_context()
            self.sim.physics_manager.set_debug_context_provider(
                "curriculum_level", lambda: self.curriculum_level
            )

Manager-based RL environments already register ``episode_length`` and
``common_step_counter``. Providers are resolved when a snapshot is taken, must
return a non-``None`` value, and must keep the initialization-time schema.
Duplicate names fail unless replacement is explicitly requested. Use
:meth:`~isaaclab.physics.PhysicsManager.remove_debug_context_provider` when a
manually registered provider no longer applies.

Record solver operations
------------------------

Incident archives and transition replay can include transient operations that
are not retained on the solver object. For exactly one compatible MJWarp-backed
``SolverMuJoCo``, whether direct or within a coupled mapping and running on a
Warp CPU or CUDA device, enabling either
:attr:`~isaaclab_newton.physics.NewtonDebugCaptureCfg.record_operations` or
:attr:`~isaaclab_newton.physics.NewtonDebugReplayCfg.record_operations`
automatically installs
:class:`~isaaclab_newton.physics.MJWarpDebugOperationProvider`. The built-in
provider interposes installed Python call boundaries without patching MJWarp or
producing an installed-source diff, and records the final ``SolverContext`` and
``CollisionContext``. Native ``SolverMuJoCo(use_mujoco_cpu=True)``, ambiguous
mappings, and other solvers require an explicit compatible provider. An
explicitly registered provider always wins.

Operation recording requires ``use_cuda_graph=False`` for every provider.
Python call interposition and host-side snapshots execute while a graph is
captured, not on each graph launch, so allowing this combination would retain
stale operation evidence.

When operation recording is configured, custom incident triggers receive the
same current snapshot as ``context.operations``. This supports finite but
invalid conditions such as an excessive broad-phase candidate count without
waiting for a separate state NaN.

A custom provider implements ``bind(solver)``, ``snapshot()``, and ``close()``:

.. code-block:: python

    from collections.abc import Callable

    from isaaclab_newton.physics import NewtonManager


    class OperationProvider:
        def __init__(self, snapshot_fn: Callable):
            self._snapshot_fn = snapshot_fn
            self._solver = None

        def bind(self, solver) -> None:
            self._solver = solver

        def snapshot(self):
            if self._solver is None:
                raise RuntimeError("operation provider is not bound")
            return self._snapshot_fn(self._solver)

        def close(self) -> None:
            self._solver = None


    NewtonManager.set_debug_operation_provider(
        OperationProvider(lambda solver: solver.operation_snapshot)
    )

The recorder calls ``bind(solver)`` once during physics initialization and owns
the provider after binding. ``snapshot()`` must return a non-``None``,
discoverable object whose schema remains stable across incident and replay
snapshots. ``close()`` releases hooks and solver references when the recorder
is cleared or when later initialization fails; a failed close remains owned so
cleanup can be retried. Missing methods, provider errors, and schema drift fail
loudly. Register an explicit provider before the first solver initialization;
duplicate or late registration is an error. A registered provider with both
incident and replay operation recording disabled also fails initialization. The ``operation_snapshot``
attribute above illustrates solver-specific state and is not a Newton API.

The built-in provider can also preserve the first non-finite MJWarp solver
iteration. Configure the narrow iteration scan once, then limit dispatch-time
non-finite detection to the preserved workspace while still archiving the full
operation snapshot:

.. code-block:: python

    from isaaclab_newton.physics import (
        MJWarpDebugOperationProvider,
        NewtonDebugCaptureCfg,
        NewtonDebugReplayCfg,
        NewtonManager,
    )

    NewtonManager.set_debug_operation_provider(
        MJWarpDebugOperationProvider(
            first_nonfinite_include_fields=(
                "mjwarp_solver_context.h",
                "mjwarp_solver_context.hfactor",
            ),
        )
    )

    debug_capture = NewtonDebugCaptureCfg(
        record_solver=True,
        record_operations=True,
        detect_nonfinite_in=("state", "operations"),
        detect_nonfinite_include_fields=(
            "state.*",
            "operations.first_nonfinite_context.h",
            "operations.first_nonfinite_context.hfactor",
        ),
        replay=NewtonDebugReplayCfg(
            enabled=True,
            record_state=True,
            record_solver=True,
            record_operations=True,
        ),
    )

The provider validates every configured full-path glob against the installed
MJWarp schema. Iteration scanning temporarily disables the MJWarp graph
conditional, records the matching paths, and clones ``pre_solve_data``, the
first complete non-finite ``SolverContext``, and ``first_nonfinite_data`` from
that same iteration.

Per-world attribution is emitted only when an inspected field declares explicit
``nworld`` ownership metadata. Ambiguous solver workspaces are global and remain
full in the artifact; matching a coincidental first dimension never truncates
them. A value that is born and overwritten entirely inside one compiled kernel
cannot be discovered from a host Python boundary. Retain it upstream or expose
it through a solver-specific custom hook before relying on this recorder.

:meth:`~isaaclab_newton.physics.MJWarpDebugOperationProvider.close`
restores the original function bindings and graph option. Any upstream call-path
or schema change fails during initialization instead of silently reducing the
diagnostic.

Diagnosis recipes
-----------------

Use the manifest field inventory as the source of truth for the installed
Newton version, then follow the relevant data path:

* **External collision pipeline:** Enable ``record_model``, ``record_contacts``,
  ``record_collision_pipeline``, and ``record_solver``. Start with
  ``collision_pipeline.broad_phase_pair_count`` and
  ``collision_pipeline.broad_phase_shape_pairs``; verify the corresponding
  ``collision_pipeline.narrow_phase.shape_aabb_lower`` and
  ``shape_aabb_upper``; use the manifest active-row metadata to correlate
  those shape IDs with the finalized ``contacts`` rows, then inspect the
  solver's retained constraint fields. Enable
  ``replay.record_collision_pipeline`` to compare pipeline pre/post values over
  time. The cold incident snapshot alone records
  only the failing step's finalized pipeline.
* **Internal MJWarp collision:** Leave ``record_collision_pipeline=False`` and
  enable ``record_solver``. Retained ``solver.mjw_data`` fields include
  ``nacon``, ``contact.*``, ``nefc``, ``efc.*``, ``qM``, ``qLD``, and
  ``qLDiagInv``; discovery records the fields present in the installed version
  and active-row contracts exclude unused capacity. Enable incident
  ``record_operations=True`` to capture the final MJWarp ``SolverContext`` and
  broad-phase ``CollisionContext`` alongside those retained arrays; enable the
  replay option when every transition snapshot is required.
* **Deep Hessian non-finite:** Use the configured
  ``MJWarpDebugOperationProvider`` example above to scan only ``h`` and
  ``hfactor`` after each solver iteration and preserve the first complete
  failing context. Set ``capture_per_substep=True`` and
  ``use_cuda_graph=False`` when a dispatch contains multiple solver substeps;
  otherwise a later substep can replace the saved operation snapshot before
  dispatch-post detection. The recorder scan filters keep the trigger narrow
  while the artifact retains the full operation evidence.
* **Finite joint-velocity spike:** Register the ``joint_velocity_spike``
  trigger above. Its returned ``world_ids`` create focused artifacts even
  though ``joint_qd`` contains no NaN or infinity.
* **Failure immediately before a reset:** The synchronous dispatch-post or
  solver-post export freezes the failing state, solver, contacts, and workflow
  context before the next task reset can mutate them. Compare ``pre__*`` with
  ``history__*`` or ``replay__pre__*`` with ``replay__post__*``; keep
  ``solver_reset`` enabled independently to clear solver-owned state on the
  following task-authored reset.

Inspect, validate, compare, and replay
--------------------------------------

Use the unified command-line tool through the Isaac Lab Python wrapper:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/physics_debug.py inspect physics_incident_....npz
    ./isaaclab.sh -p scripts/tools/physics_debug.py inspect physics_incident_....npz --json
    ./isaaclab.sh -p scripts/tools/physics_debug.py validate physics_incident_....npz
    ./isaaclab.sh -p scripts/tools/physics_debug.py validate physics_incident_....npz \
        --required_key history__state__joint_q
    ./isaaclab.sh -p scripts/tools/physics_debug.py validate physics_incident_....npz \
        --allowed_status partial
    ./isaaclab.sh -p scripts/tools/physics_debug.py diff left.npz right.npz
    ./isaaclab.sh -p scripts/tools/physics_debug.py replay physics_incident_....npz \
        --stage transition
    ./isaaclab.sh -p scripts/tools/physics_debug.py replay physics_incident_....npz \
        --adapter_module my_project.physics_debug_adapters \
        --allowed_status partial --capability transition_history --json

``validate`` accepts only complete archives by default; partial archives require
an explicit ``--allowed_status partial``. ``diff`` compares manifest schema,
dependency and runtime provenance, and array bytes exactly and exits with
status 1 for a mismatch. All commands reject checksum, dtype, shape, or
manifest tampering and load arrays without pickle.

Replay also accepts only complete archives by default.
``--allowed_status partial`` is an explicit archive-level opt-in for cases
such as an unrelated scene-export failure. It does not weaken capability validation: the selected
capability must still be declared complete with every required field and a
registered adapter.

Replay is capability-driven. A module supplied with ``--adapter_module`` is
explicitly trusted application code and must call ``register_replay_adapter``
when imported. Repeat the option when adapters span multiple modules. The tool
never imports an archive-controlled module or guesses behavior from old flat
keys. An empty, duplicate, or unimportable module name, incomplete capability,
missing required field, missing adapter, or ambiguous stage is an actionable
error; use ``--capability`` to select one exact declaration.

Adapter modules use the package-owned registry, including when the CLI runs
directly as a script:

.. code-block:: python

    from isaaclab_newton.physics.debug_replay import (
        ReplayAdapter,
        ReplayRequest,
        register_replay_adapter,
    )


    def replay_state(request: ReplayRequest):
        return {"body_count": int(request.arrays["history__state__body_q"].shape[1])}


    register_replay_adapter(
        ReplayAdapter(
            adapter_id="my_project.state.v1",
            stages=frozenset({"state"}),
            providers=frozenset({"isaaclab_newton.physics_incident_recorder"}),
            required_fields=("history__state__body_q",),
            callback=replay_state,
        )
    )
