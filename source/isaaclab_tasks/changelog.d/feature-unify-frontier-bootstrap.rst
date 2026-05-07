Fixed
^^^^^

* Restored a tiny non-zero ``eps`` (``1e-8``, matching the non-frontier presets) in all terrain and factory ``frontier*`` curriculum presets. With ``eps=0`` plus ``BetaSignal`` returning 0 at ``s=0``, the per-state probability vector at training step 0 (before any successes have been observed) summed to zero, tripping ``torch.multinomial``'s ``cumdist[size - 1] > 0`` device-side assert during the first ``env.reset()``. The new floor is small enough to leave the steady-state distribution untouched (its total mass is five orders of magnitude below the active frontier band) but enough to make the categorical valid at bootstrap.
