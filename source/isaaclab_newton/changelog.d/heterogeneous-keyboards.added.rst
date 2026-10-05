* Added context-owned exact-population and growable-world Newton backends. Growable worlds use
  prepared MuJoCo prototypes, GPU Components instance handles and virtual backing, and Newton's
  free ``mujoco_worlds_*`` operations for capture, admission, and retirement.
* Added world-masked model-property updates, prepared-constant reuse, runtime sleep policies, and
  selected-world forward-kinematics invalidation for task-owned reset writes.
* Captured physics on a private stream while preserving caller ordering, including callers using
  the default CUDA stream. Builder preparation follows the configured solver factory.
