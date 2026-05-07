Changed
^^^^^^^

* **Breaking:** Removed ``eps`` from :class:`~isaaclab_tasks.manager_based.multi_task.curriculum.BetaSignalCfg`. The Beta signal score is now the pure ``s^(a-1) * (1-s)^(b-1)`` Beta kernel and returns 0 at the s=0 / s=1 boundaries (safe for the supported ``target ∈ [0, 1]``, ``kappa ≥ 0`` range). All probability-floor semantics now route through the single :attr:`~isaaclab_tasks.manager_based.multi_task.curriculum.CurriculumCfg.eps`. Migration: drop the ``eps=...`` argument from any ``BetaSignalCfg(...)`` constructor call; tune the curriculum-level ``eps`` if you relied on the per-signal floor for boundary mass.
