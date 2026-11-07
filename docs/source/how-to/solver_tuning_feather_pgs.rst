:orphan:

.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _feather-pgs-solver-tuning:

FeatherPGS Solver
=================

FeatherPGS is an experimental Newton solver for articulated rigid-body systems.
It evaluates reduced-coordinate dynamics and resolves contact and enabled joint
constraints with projected Gauss-Seidel (PGS) iterations. Isaac Lab exposes it
through :class:`~isaaclab_newton.physics.FeatherPGSSolverCfg`, which selects
:class:`~isaaclab_newton.physics.FeatherPGSSolver`.

FeatherPGS uses Newton's :class:`CollisionPipeline`. Configure contact
generation with
:class:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg`, as with the
Featherstone and XPBD Newton solvers.

Configuration
-------------

Select FeatherPGS by assigning its solver configuration to
:attr:`~isaaclab_newton.physics.NewtonCfg.solver_cfg`:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import (
        FeatherPGSSolverCfg,
        NewtonCfg,
        NewtonCollisionPipelineCfg,
    )

    solver_cfg = FeatherPGSSolverCfg(
        enable_joint_limits=True,
        pgs_iterations=8,
        dense_max_constraints=64,
        mf_max_constraints=512,
        pgs_beta=0.05,
        pgs_cfm=1.0e-6,
    )
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        collision_cfg=NewtonCollisionPipelineCfg(),
        num_substeps=1,
    )
    sim_cfg = SimulationCfg(dt=1.0 / 200.0, physics=newton_cfg)

The important accuracy and runtime controls are:

* :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.pgs_iterations`:
  increases constraint-solve work and usually improves constraint fidelity.
* :attr:`~isaaclab_newton.physics.NewtonCfg.num_substeps`: repeats collision,
  integration, and the solve at a shorter effective time step.
* :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.dense_max_constraints`
  and
  :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.mf_max_constraints`:
  preallocate per-world constraint capacity. Undersizing either capacity can
  omit constraints; oversizing it consumes memory and can reduce throughput.
* :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.pgs_beta` and
  :attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.pgs_cfm`: control
  position correction and solver regularization.

Solve Formulation
-----------------

FeatherPGS chooses its formulation from the device. On CUDA it solves every
model matrix-free: it never assembles the Delassus matrix, and its sweeps read
each constraint row's response from the factored mass matrix. On the CPU it
assembles the Delassus rows and sweeps them one thread per world.

Kernels, factors and buffers follow from the device and the model's topology.
See :ref:`feather-pgs-automatic-selection` for the collision choices that remain. Set
:attr:`~isaaclab_newton.physics.FeatherPGSDiagnosticsCfg.nvtx` of
:attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.diagnostics` to ``True``
to emit solver-stage ranges when profiling.

Current Limitations
-------------------

FeatherPGS support is experimental. Supported task families expose the
``feather_pgs`` preset from their existing physics preset container. Robot
variants inherit that family-level solver, contact-capacity, and substep
configuration without defining backend-specific physics or event subclasses:

.. code-block:: bash

    uv run isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 presets=feather_pgs

Tasks that do not declare this preset have not been validated with FeatherPGS.

The current Newton implementation also has these restrictions:

* Floating-base systems require an explicit free joint connecting the root body
  to the world.
* Joint velocity-limit constraints require a CUDA device.


.. _feather-pgs-automatic-selection:

Automatic Selection
-------------------

FeatherPGS chooses its factor, kernels, streams and buffers from the device and
the model's topology. Only the collision choices below are configurable. Leave
them at ``auto`` for normal use, and override only the one you want to pin.

.. code-block:: python

    from isaaclab_newton.physics import FeatherPGSSolverCfg

    cfg = FeatherPGSSolverCfg()  # automatic collision choices
    cfg.convex_collision = "ordinary"  # force ordinary convex queries

    # After simulation initialization, outside CUDA graph capture:
    choices = sim.backend_manager.backend.solver_ops.path_selection()

.. list-table:: Collision choices
   :header-rows: 1
   :widths: 28 32 40

   * - Setting
     - Choices
     - Automatic behavior
   * - ``convex_collision``
     - ``auto``, ``ordinary``, ``rejection``, ``convex_cells``
     - Predicted cost chooses ordinary queries, rejection screening or convex cells; unsupported pipelines use ordinary.
   * - ``contact_export``
     - ``auto``, ``tiled``, ``compact``
     - Uses compact reduced-contact export for supported CUDA mesh/heightfield pipelines.

An unsupported forced choice raises an error. Selection does not change
timestep, iteration budgets, friction coefficients or the solver formulation.

The collision selectors are experimental. Automatic collision includes convex
cells where immutable hull geometry supports them. Cells reduce support
searches in screening, queries and contact generation; the selector includes
descriptor, fallback and dispatch costs. Calibration uses independent synthetic
primitives and samples current geometric work.
Cell tables and calibration are built before graph capture; they add setup
time. Selection can change during replay. Call
``CollisionPipeline.invalidate_convex_cells()`` before modifying mesh geometry.
Existing graphs then use ordinary support scans; rebuild the pipeline to
enable cells for the changed geometry.

Contact export uses structural admission rules. Convex collision uses the
sampled cost model described here.
Automatic compact export excludes deterministic, speculative, hydroelastic and
alternate heightfield export modes; these retain their existing exporter.
Inspect ``choices["collision"]["contact_export"]`` for the selected mapping.

Migration
---------

The fields that pinned the solver's implementation are removed, because
FeatherPGS now chooses it from the device and the model: ``sparse_mass_matrix``,
``cached_dynamics``, ``cholesky_kernel``, ``trisolve_kernel``,
``hinv_jt_kernel``, ``delassus_kernel``, ``pgs_kernel``, ``pgs_chunk_size``,
``use_parallel_streams``, ``double_buffer`` and ``tile_threads``. Delete them
from configurations.

``convex_collision`` overrides ``NEWTON_NARROW_PHASE_COHERENT_CONVEX`` and
``NEWTON_NARROW_PHASE_CONVEX_CELLS`` for FeatherPGS. Move an existing manual
cell recipe to ``convex_collision="convex_cells"``. Direct Newton callers can
pass the same choice to ``CollisionPipeline``; omitting that argument retains
the legacy environment behavior. Recreate the solver/pipeline after changing
a configuration choice. Collision inspection reads the latest device choice
and must run outside capture.

``contact_export`` overrides ``NEWTON_REDUCER_COMPACT_EXPORT``. Use ``tiled`` to
retain the original export mapping, or ``compact`` to require compact admission.
Direct Newton callers can pass the same choice to ``CollisionPipeline``;
omitting it preserves legacy environment selection.
