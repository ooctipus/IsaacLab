Changed
^^^^^^^

* :func:`~isaaclab_tasks.manager_based.multi_task.terrain.mdp.commands.task_table_builder.build_task_table` now thins via :func:`~isaaclab_tasks.manager_based.multi_task.terrain.retarget.apply_final_fps` directly instead of routing through a one-shot :class:`~isaaclab_tasks.manager_based.multi_task.curriculum.StateBuffer`. ``StateBuffer`` retains its native role as the streaming factory accumulator's container; the locomotion task table is one-shot and gets the lighter helper. Both still share ``grid_bucket_downsample`` and the ``extract_features`` dispatch, so primitives stay unified — only the container is per-lifecycle now. Memory peak at the task-table-build call drops by roughly 7× at high candidate counts (no zero-allocated buffer slab + no add/clone copies).
