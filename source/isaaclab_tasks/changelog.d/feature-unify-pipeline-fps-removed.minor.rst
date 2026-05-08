Changed
^^^^^^^

* **Breaking:** :meth:`RetargetPipeline.run` no longer runs the terminal FPS spatial-thinning step internally. It always returns ``buffer._selected[:num_selected]`` populated with every post-criteria candidate (what ``skip_final_fps=True`` did in the prior step). Callers that want the thin run :func:`~isaaclab_tasks.manager_based.multi_task.terrain.retarget.apply_final_fps` afterwards (one-shot scripts) or feed the candidates through :class:`~isaaclab_tasks.manager_based.multi_task.curriculum.StateBuffer.compact` (streaming buffers). Migration: drop any ``skip_final_fps`` reads/writes (the cfg field is gone), and add an ``apply_final_fps(buffer, n_desired, extractor=..., spacing=...)`` call after ``pipeline.run`` if you previously relied on the post-FPS subset.

Added
^^^^^

* Added :func:`~isaaclab_tasks.manager_based.multi_task.terrain.retarget.apply_final_fps` and :func:`~isaaclab_tasks.manager_based.multi_task.terrain.retarget.bbox_target_count` helpers in ``feature_extractors``. ``apply_final_fps`` is the in-place rewrite of ``buffer._selected`` / ``num_selected`` that the pipeline used to do internally, lifted out so any caller can trigger the thin without re-implementing it. ``bbox_target_count`` is the spacing → count derivation, factored out so the formula has one home.

Removed
^^^^^^^

* Removed the ``RetargetPipelineCfg.skip_final_fps`` flag introduced in the previous step. The flag's purpose -- giving callers an opt-in to skip the in-pipeline thin -- is now the only behaviour, so the gate has no remaining function. Migration: delete ``.replace(skip_final_fps=True)`` calls and any reads of the field.
