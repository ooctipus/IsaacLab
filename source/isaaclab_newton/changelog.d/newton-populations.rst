Added
^^^^^

* Added an experimental headless Newton population backend owned by ``SimulationContext``.
  It replicated prepared native-contact MuJoCo Warp prototypes into exact homogeneous
  populations and retained unchanged models, solver state, controls, and graphs.
  Resizing transferred surviving worlds' physical properties, solver history, and
  contacts on the GPU before publishing replacements.
* Added partial resets and property updates over bounded population streams, with
  producer and completion events preserving caller ordering. Captured resets preserved
  task-written joint coordinates. Construction validated timestep and graph cadence,
  and reset masks were validated before mutation. Rendering remained unsupported.
