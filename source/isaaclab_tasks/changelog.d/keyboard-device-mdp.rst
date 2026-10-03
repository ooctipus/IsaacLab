Changed
^^^^^^^

* Reduced temporary tensor work in keyboard observations, typing updates, snapshot restoration and success monitoring.
* Exposed read-only selected pose fields and explicit dense shapes, allowing pose observations to read native storage directly.
* Allowed success-monitor updates to exclude outcomes with an explicit validity mask while preserving untouched slot priors.
* Used monotone memory growth for keyboard reset batches that only required additional capacity.
* Retried budget-limited reset admission once without optional backing headroom, preserving all live source and incoming worlds before publication.
* Skipped unchanged host rendering descriptors during keyboard resets while preserving all physical property restoration.
* Corrected ordinary keyboard terminal observations to include the continuing typing update and successor observation history, matching the growable-world task without advancing live command state or RNG.
* Declared keyboard typing commands as reset-only, removing the unused ten-second resampling timer from six-second episodes. Removing timer random draws changed subsequent seeded sampling; finite timer configurations retained their existing behavior.
* Ordered keyboard task actions, physics, resets and observations on the caller's Torch stream, including when it differed from Warp's current stream.
