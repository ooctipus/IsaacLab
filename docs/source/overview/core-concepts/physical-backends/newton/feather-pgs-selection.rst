FeatherPGS automatic selection
=============================

FeatherPGS chooses its factor, kernels, streams and buffers from the device and
the model's topology. Only the collision choices below are configurable. Leave
them at ``auto`` for normal use, and override only the one you want to pin.

.. code-block:: python

    from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonFeatherPGSManager

    cfg = FeatherPGSSolverCfg()  # automatic collision choices
    cfg.convex_collision = "ordinary"  # force ordinary convex queries

    # After simulation initialization, outside CUDA graph capture:
    choices = NewtonFeatherPGSManager.get_path_selection()

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
