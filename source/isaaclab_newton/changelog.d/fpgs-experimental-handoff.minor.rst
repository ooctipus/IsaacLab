Added
^^^^^

* Added experimental, opt-in FeatherPGS grouped dynamics, lazy kinematics publication, and alternative
  constraint sweep controls for profiling. The parallel projection sweep did not preserve the legacy
  solver's converged velocities; keep ``mf_gs_parallel_rows=0`` when legacy solver semantics are required.
* Added experimental collision profiling controls through ``NEWTON_COLLISION_BOX_SAT``,
  ``NEWTON_CONTACT_MATCHING``, and ``NEWTON_NARROW_PHASE_THREADS_X`` environment variables.
