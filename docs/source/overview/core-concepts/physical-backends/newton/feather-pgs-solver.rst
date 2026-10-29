FeatherPGS Solver
=================

FeatherPGS is an experimental Newton solver for articulated rigid-body systems.
It evaluates reduced-coordinate dynamics and resolves contact and enabled joint
constraints with projected Gauss-Seidel (PGS) iterations. Isaac Lab exposes it
through :class:`~isaaclab_newton.physics.FeatherPGSSolverCfg` and
:class:`~isaaclab_newton.physics.NewtonFeatherPGSManager`.

FeatherPGS uses Newton's :class:`CollisionPipeline`. Configure contact
generation with
:class:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg`, as with the
Featherstone and XPBD Newton managers.

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
See :doc:`feather-pgs-selection` for the collision choices that remain. Set
:attr:`~isaaclab_newton.physics.FeatherPGSSolverCfg.nvtx` to ``True`` to
emit solver-stage ranges when profiling.

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
* Joint velocity-limit constraints, schedules other than ``"interleaved"``,
  velocity iterations and ``drive_mode="physx_pgs"`` require a CUDA device.
