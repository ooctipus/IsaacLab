Added
^^^^^

* Added the experimental headless ``IsaacContrib-Keyboard-SO101-Populations`` task.
  It ran the shared typing MDP and curriculum over exact Newton/MJWarp populations,
  queued desired keyboards at reset, and reconciled assignments at a configurable cadence.
  Task-local selections bound explicit model/state/control resources; padding remained
  only in policy observations and reset snapshots. Surviving worlds preserved native
  state, solver history, and contacts; unchanged counts retained their models and graphs.
* Added explicit administrative truncation at the default 128-control-step redistribution
  boundary, excluding interrupted episodes from curriculum success/failure statistics.
  Timeouts bootstrapped from pre-reset successor observations containing the current
  typing update and history sample, while preserving reward, reset, and RNG state.
  True terminations retained priority, and finite-horizon administrative truncation was rejected.

Changed
^^^^^^^

* Encoded target and typed key slots directly into float observations, preserving padding
  and active-key masks while avoiding intermediate integer one-hot buffers.
* Fused the shared relative-PD action update, preserving target refresh, participation
  masks, and effort telemetry. Recorded selected forward kinematics and Jacobians once
  per reset IK solve, followed by one native reconciliation. Independent population
  resets used bounded concurrent streams with explicit producer and completion ordering.
* Retired replaced selection caches and task-owned bindings explicitly, including startup
  and cleanup failure paths. External selections retained valid native resources without
  hot-path garbage collection. Failed binding publication blocked further task operations
  while preserving the original error and allowing idempotent cleanup.
