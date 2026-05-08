Added
^^^^^

* Added oversample-then-FPS-thin to :class:`~isaaclab_tasks.manager_based.multi_task.curriculum.StateBufferCfg` via two new fields, ``oversample_ratio`` and ``fps_features``, mirroring the locomotion retarget pipeline's ``final_fps_features`` API. When ``oversample_ratio > 1.0`` the buffer accumulates ``size * oversample_ratio`` states then compacts back down to ``size`` via :func:`~isaaclab_tasks.manager_based.multi_task.grid_downsample.grid_bucket_downsample`, preserving spatial diversity rather than relying on FIFO eviction. The default ``oversample_ratio = 1.0`` reproduces the legacy ring-buffer-with-FIFO-wrap behaviour. Compaction notifies registered callbacks so callers can permute parallel arrays (success rate, monitor history) in lockstep with the buffer.
