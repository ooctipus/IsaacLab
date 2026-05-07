Changed
^^^^^^^

* Mirrored the terrain frontier-preset retune in factory: bumped ``FrontierSignalCfg.dilation_steps`` from 1 to 2 and dropped the curriculum-level ``eps`` from 1e-3 to 0.0 across the six factory ``frontier*`` presets. Same motivation as the terrain change -- previous configuration was giving most of the sampling mass to already-mastered states via ``eps``.
