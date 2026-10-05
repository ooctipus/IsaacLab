* Added explicit action targets, reset-only command scheduling, and successor-observation previews
  that preserve command state, observation history, and random-number generators. Custom preview
  functions must be pure; unsupported stateful callbacks are rejected before evaluation.
* Added a final-observation hook and a shared Torch/Warp stream scope for task-owned physics and MDP
  operations. Added scene-data geometry revisions for reset-time mesh and visibility changes.
* Added W&B run-name checkpoint lookup for cluster launches and local asset-path checks that avoid
  initializing the Omniverse client for missing absolute paths.
