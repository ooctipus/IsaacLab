Fixed
^^^^^

* Skipped Newton solver-reset and forward-kinematics work at clean state
  boundaries while preserving reset-mask writes replayed from CUDA graphs.
