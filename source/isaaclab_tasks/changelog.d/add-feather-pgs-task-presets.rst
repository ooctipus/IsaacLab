Added
^^^^^

* Added the ``feather_pgs`` physics preset to supported task-family composition
  roots for the Newton FeatherPGS solver.

Changed
^^^^^^^

* Changed velocity tasks to share domain randomization, actuator conditioning,
  and PPO iteration budgets across physics solvers.

Fixed
^^^^^

* Fixed excessive FeatherPGS work in dexterous-hand tasks and unnecessary
  joint-limit row allocation in cabinet tasks.
* Fixed dexterous-hand FeatherPGS constraint truncation while retaining the
  shared simulation cadence.
