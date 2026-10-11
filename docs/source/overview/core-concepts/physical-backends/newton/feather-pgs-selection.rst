FeatherPGS automatic selection
=============================

Leave implementation choices at ``auto`` for normal use. Override only the path
you want to pin; the remaining choices stay automatic.

.. code-block:: python

    from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonFeatherPGSManager

    cfg = FeatherPGSSolverCfg()  # automatic collision, factors and kernels
    cfg.convex_collision = "ordinary"  # force ordinary convex queries

    # After simulation initialization, outside CUDA graph capture:
    choices = NewtonFeatherPGSManager.get_path_selection()

.. list-table:: Implementation choices
   :header-rows: 1
   :widths: 28 32 40

   * - Setting
     - Choices
     - Automatic behavior
   * - ``convex_collision``
     - ``auto``, ``ordinary``, ``rejection``, ``convex_cells``
     - Predicted cost chooses ordinary queries, rejection screening or convex cells; unsupported pipelines use ordinary.
   * - ``sparse_mass_matrix``
     - ``auto``, ``on``, ``off``
     - Uses native sparse factors where topology, solver policy and device resources permit.
   * - ``contact_export``
     - ``auto``, ``tiled``, ``compact``
     - Uses compact reduced-contact export for supported CUDA mesh/heightfield pipelines.
   * - ``cached_dynamics``
     - ``auto``, ``on``, ``off``
     - Pairs cached prediction with next-state publication where the topology and solver policy permit.
   * - ``cholesky_kernel`` / ``trisolve_kernel``
     - ``auto``, ``loop``, ``tiled``
     - Uses existing size and device rules.
   * - ``hinv_jt_kernel``
     - ``auto``, ``par_row``, ``tiled``
     - Uses existing topology, size and memory rules.
   * - ``delassus_kernel``
     - ``auto``, ``par_row_col``, ``tiled``
     - Selects dense assembly for split mode.
   * - ``pgs_kernel``
     - ``auto``, ``loop``, ``tiled_row``, ``tiled_contact``, ``streaming``
     - Uses scalar CPU or tiled CUDA sweeps in split mode; matrix-free owns its sweep dispatcher.

An unsupported forced choice raises an error. Explicit dense kernels disable
competing automatic sparse/paired implementations. Use ``auto`` for dense PGS
and Delassus selectors in matrix-free mode, where those stages are unused.
Selection does not change timestep, iteration budgets, friction coefficients
or the selected solver formulation.

Collision and native sparse-factor selectors are experimental. Automatic
collision includes convex cells where immutable hull geometry supports them.
Cells reduce support searches in screening, queries and contact generation;
the selector includes descriptor, fallback and dispatch costs. Calibration
uses independent synthetic primitives and samples current geometric work.
Cell tables and calibration are built before graph capture; they add setup
time. Selection can change during replay. Call
``CollisionPipeline.invalidate_convex_cells()`` before modifying mesh geometry.
Existing graphs then use ordinary support scans; rebuild the pipeline to
enable cells for the changed geometry.

Cached prediction, native sparse composition and contact export use structural
admission rules. Convex collision uses the sampled cost model described here.
Automatic compact export excludes deterministic, speculative, hydroelastic and
alternate heightfield export modes; these retain their existing exporter.
Inspect ``choices["collision"]["contact_export"]`` for the selected mapping.

Migration
---------

The four kernel fields now reach Newton rather than being silently filtered
out. Unsupported explicit values therefore produce errors. Use ``auto`` when
the same configuration must run on CPU and CUDA.

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
