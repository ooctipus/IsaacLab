Changed
^^^^^^^

* **Breaking:** Renamed ``SamplerSizingCfg.final_fps_features`` → ``fps_features`` and ``final_fps_spacing`` → ``fps_spacing`` so the field names match :class:`~isaaclab_tasks.manager_based.multi_task.curriculum.StateBufferCfg.fps_features` (the same concept, two consumers, one name now). ``final_fps_oversample`` (a sampler-sizing back-cascade multiplier, distinct concept) keeps its name. Migration: rename your cfg keys; ``preset(final_fps_features=...)`` constructions in ``command_presets.py`` are an example.

Added
^^^^^

* Added :func:`~isaaclab_tasks.manager_based.multi_task.utils.grid_downsample.extract_features`, a shared dispatch helper for ``(states, extractor) -> features``. Both :func:`~isaaclab_tasks.manager_based.multi_task.terrain.retarget.apply_final_fps` and :class:`~isaaclab_tasks.manager_based.multi_task.curriculum.StateBuffer.compact` now call it instead of duplicating the four-line dispatch.
* Added :meth:`~isaaclab_tasks.manager_based.multi_task.terrain.retarget.RetargetBuffer.set_selected`, mediating the ``_selected`` / ``num_selected`` write so FPS-thinning helpers don't poke private fields directly.

Fixed
^^^^^

* Cut the transient-memory peak of :func:`~isaaclab_tasks.manager_based.multi_task.terrain.retarget.apply_final_fps` for the default xyz extractor by skipping the full-slab gather (was ``[N, state_dim]``, now just ``[N, 3]``). At ``N = 1M`` candidates with ``state_dim ≈ 25`` this saves ~88 MB of transient allocation. Custom extractors still receive the full slab.
* Fixed :func:`~isaaclab_tasks.manager_based.multi_task.terrain.retarget.bbox_target_count` docstring claim that the return is "``>= 1``"; the empty-input case correctly returns ``0``.
