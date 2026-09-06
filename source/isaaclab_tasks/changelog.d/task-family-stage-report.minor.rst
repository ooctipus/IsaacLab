Added
^^^^^

* Added per-stage construction reports to task-family execution.
  :func:`~isaaclab_tasks.core.multi_task.mdp.commands.state_command.execute_task_family` now
  times every generate, solve, criterion, and selection stage, records the candidate rows
  entering and leaving each one, and returns the result as
  :class:`~isaaclab_tasks.core.multi_task.mdp.commands.state_command.TaskFamilyReport` on
  ``TaskFamilyExecution.report``. Stage functions attach domain facts with
  :func:`~isaaclab_tasks.core.multi_task.mdp.commands.state_command.record_stage_details`;
  the Position and Factory solve stages report IK batch statistics and the Position generator
  reports sampler rejection counts. The INFO log emitted during inspection prints the report
  with one aligned line per stage instead of a single counts line.
