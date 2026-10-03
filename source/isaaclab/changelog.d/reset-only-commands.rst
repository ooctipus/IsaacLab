Added
^^^^^

* Added explicit reset-only command scheduling through ``CommandTermCfg.resampling_time_range=None``. Timed scheduling remained unchanged. Reset-only scheduling skipped timer updates, expiry queries and timer random draws; opting in can therefore change subsequent seeded random samples. Explicit resets still sampled commands and reset their counters.
