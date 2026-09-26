.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Newton Native Data and Selection API
====================================

Mental model
------------

The Newton backend exposes its engine-owned `Model
<https://newton-physics.github.io/newton/stable/api/_generated/newton.Model.html>`_, `State
<https://newton-physics.github.io/newton/stable/api/_generated/newton.State.html>`_, `Control
<https://newton-physics.github.io/newton/stable/api/_generated/newton.Control.html>`_, and optional
``Contacts`` objects through :class:`isaaclab_newton.physics.NewtonManager`. Their Warp arrays are
the live engine data, rather than values pulled into a separate per-asset view buffer. The model
owns structural and static arrays and labels; a state owns evolving simulation arrays; and a
control owns actuation inputs. Contacts are optional and depend on the active solver and collision
path.

Lifecycle prerequisite
----------------------

Access Newton data after the simulation context has initialized the physics backend and built its
model. Treat the objects as invalid after a model rebuild and reacquire them after Isaac Lab has
reinitialized the simulation. This API is intended for code that can own the necessary lifecycle
and synchronization responsibilities.

Reuse Isaac Lab-owned access
----------------------------

Use the manager accessors to obtain the current Newton objects:

.. code-block:: python

   from isaaclab_newton.physics import NewtonManager

   model = NewtonManager.get_model()
   state = NewtonManager.get_state_0()
   control = NewtonManager.get_control()
   contacts = NewtonManager.get_contacts()

   body_poses = state.body_q
   joint_forces = control.joint_f

:meth:`isaaclab_newton.physics.NewtonManager.get_model` can construct a visualization shadow
model when PhysX is active. The write semantics in this guide apply only when Newton is the active,
authoritative physics backend.

Isaac Lab's Newton-backed assets expose the same generic
``newton.selection.ArticulationView`` selection helper. For example, reuse an articulation's
root selection instead of constructing the matching selection again:

.. code-block:: python

   robot = scene["robot"]
   selection = robot.root_view
   joint_positions = selection.get_dof_positions(state)

Newton uses this generic, label-based selection concept for Isaac Lab articulations, rigid objects,
rigid-object collections, and cables. It is a selection helper over model indices, not per-asset
typed storage.

Create raw access
-----------------

Code that owns a matching model can construct its own selection from a model and a label pattern:

.. code-block:: python

   from newton.selection import ArticulationView

   selection = ArticulationView(
       model,
       pattern="/World/envs/env_*/Robot",
   )

Read/write semantics
--------------------

Selections provide typed convenience methods as well as generic string-keyed
``get_attribute()`` and ``set_attribute()`` methods. The generic methods expose
engine properties that do not have dedicated selection methods. Clone a
selected value before modifying it when you want the write to remain explicit:

.. code-block:: python

   import warp as wp
   from newton import ModelFlags
   from isaaclab_newton.physics import NewtonManager

   rolling_friction = wp.clone(
       selection.get_attribute("shape_material_mu_rolling", model)
   )
   # Modify rolling_friction with a Warp kernel before writing it back.
   selection.set_attribute(
       "shape_material_mu_rolling",
       model,
       rolling_friction,
   )
   NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

The string names a Newton model attribute rather than an Isaac Lab field. This
example uses rolling friction because it has no dedicated selection method.
Notify the manager with the flag appropriate to the property changed; other
model writes can require a different flag. State writes that change generalized
coordinates instead require forward-kinematics synchronization through
``NewtonManager.invalidate_fk()`` and ``NewtonManager.forward()``.

Ownership, synchronization, and invalidation
---------------------------------------------

Direct writes bypass Isaac Lab caches and shape/order validation. Some Newton solvers swap current
and next state buffers, so reacquire
:meth:`isaaclab_newton.physics.NewtonManager.get_state_0` when code needs the current authoritative
state on a later step. A selection can survive state-buffer swaps because it describes model
indices, but it must be recreated after a model rebuild. Solver-specific generalized- and
maximal-coordinate conventions remain authoritative.

Authoritative references
------------------------

* `Model reference <https://newton-physics.github.io/newton/stable/api/_generated/newton.Model.html>`_
* `State reference <https://newton-physics.github.io/newton/stable/api/_generated/newton.State.html>`_
* `Control reference <https://newton-physics.github.io/newton/stable/api/_generated/newton.Control.html>`_
* `ArticulationView selection reference <https://newton-physics.github.io/newton/stable/api/_generated/newton.selection.ArticulationView.html>`_
* `Newton articulation guide <https://newton-physics.github.io/newton/stable/concepts/articulations.html>`_

Task-local selections: SO101 keyboard
------------------------------------

``IsaacContrib-Keyboard-SO101`` authors its robot and 108-key keyboard with
``AssetBaseCfg`` and binds its own selectors in ``SO101KeyboardEnv.load_managers``.
The source configuration stays declarative; bindings belong to one finalized model.
Actions, MDP terms, reset snapshots, and selected fingertip Jacobians read Newton arrays
without creating an ``ArticulationView``. The baseline keeps one keyboard articulation
per world and uses MJWarp with Newton-generated contacts.

For example, the task composes a joint observation and relative key-position observation as follows:

.. code-block:: python

   from isaaclab.managers import ObservationTermCfg
   from isaaclab_tasks.contrib.keyboard import mdp
   from isaaclab_tasks.contrib.keyboard.newton_selection import BODY, JOINT_COORD, NewtonSelectorCfg

   joint_positions = ObservationTermCfg(
       func=mdp.joint_pos,
       params={"joints": NewtonSelectorCfg(JOINT_COORD, path=".*/Robot/joints/.*", count_per_world=6)},
   )
   key_positions = ObservationTermCfg(
       func=mdp.key_positions_b,
       params={
           "keys": NewtonSelectorCfg(BODY, path=".*/Keyboard/keys/key_.*", count_per_world=108),
           "root": NewtonSelectorCfg(BODY, path=".*/Robot/base", count_per_world=1),
       },
   )

Patterns full-match Newton labels. Ordered pattern lists preserve pattern order and then model
order within each world; overlapping matches are deduplicated. Coordinates and DOFs are separate
frequencies: a free joint expands to seven coordinates but six DOFs. ``count_per_world`` validates
static cardinality before episode filtering. Global entities (world -1) are excluded.

A resolved selection exposes compact ``freq_ids``, ``env_ids``, ``slot_ids``, and ``world_start``
Warp arrays. Only entries before ``world_start[-1]`` are valid. Empty worlds have equal adjacent
offsets. Storage addresses remain stable during ``env.selections.refresh()``, which rebuilds the
compact indices after changing the task's ``body_active`` or ``world_active`` masks. Dense gathers
are an explicit, uniform policy boundary and zero excluded slots. The selectors cache indices,
not state: obtain the current state with ``NewtonManager.get_state()`` at each read boundary.

Episode participation is independent of automatic physics sleeping. Naturally sleeping active
keys remain valid observations and targets. The 108-key baseline starts with every key active;
selector masking alone does not disable physics or hide geometry. Partition sleep and rendering
visibility must be driven from the same membership source when adding smaller keyboards.

After raw joint writes, call ``NewtonManager.invalidate_fk(env_ids=...)`` with int32 world IDs,
or pass an ``env_mask``. No articulation view mapping is required. This marks all articulations
in the selected worlds for the next forward/state/render boundary. For fixed-root pose changes,
write the appropriate model joint frames and call ``NewtonManager.notify_model_changed`` with
``ModelFlags.JOINT_PROPERTIES`` and a ``(world_count + 1,)`` world mask before invalidating FK.
The final mask entry represents global entities and normally remains false for task resets.

The action term preserves the original relative target and implicit-PD effort telemetry used by
the power reward. The latter is an estimate, not the solver's ``mujoco:qfrc_actuator`` output.
Reset snapshots store root poses relative to world origins followed by selected coordinates and
velocities; their layout is task-local and is not an external checkpoint format.
