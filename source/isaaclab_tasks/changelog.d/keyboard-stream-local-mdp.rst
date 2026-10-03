Changed
^^^^^^^

* Removed whole-device waits from keyboard candidate sampling and normal reset IK, preserving ordering through the task's shared Torch/Warp stream boundary.
