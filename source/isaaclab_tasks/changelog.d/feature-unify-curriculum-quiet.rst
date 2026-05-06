Added
^^^^^

* Surfaced the IK chunk count + chunk size in the task-table builder's stdout output so production runs show whether chunking actually triggered.

Changed
^^^^^^^

* Stripped the per-step ``[CURRICULUM DIAG]`` stdout block from :func:`~isaaclab_tasks.manager_based.multi_task.curriculum.log_curriculum_bins`. The same data still lands in ``log_dict`` under ``Curriculum/<signal>/{mean,p90}`` and ``Frontier/bin_*`` keys, so wandb dashboards keep working unchanged.
