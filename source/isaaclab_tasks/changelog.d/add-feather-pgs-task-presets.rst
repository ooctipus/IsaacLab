Added
^^^^^

* Added the ``feather_pgs`` physics preset to supported task-family composition
  roots for the Newton FeatherPGS solver.

Changed
^^^^^^^

* Changed velocity tasks to share domain randomization, actuator conditioning,
  and PPO iteration budgets across physics solvers.
* Changed the shared FeatherPGS velocity preset to retain two integration
  cycles and sixteen position sweeps while reusing one articulated-dynamics
  basis per physics tick.

Fixed
^^^^^

* Fixed excessive FeatherPGS work in dexterous-hand tasks and unnecessary
  joint-limit row allocation in cabinet tasks.
* Fixed dexterous-hand FeatherPGS constraint truncation while retaining the
  shared simulation cadence.
