Added
^^^^^

* Added an experimental SO101 keyboard task using prepared native world populations,
  stable GPU storage and atomic snapshot resets. Normal IK resets, snapshot replay
  and online curriculum creation shared the existing task reset implementation.
* Added task selections that read native state, controls and contact forces through
  world handles, without replicated Newton state or per-reset selection rebinding.

Fixed
^^^^^

* Corrected tapered keyboard keycap face winding so rendered surfaces faced outward.
* Preserved nested Torch CUDA stream ownership during keyboard resets and task readbacks.
