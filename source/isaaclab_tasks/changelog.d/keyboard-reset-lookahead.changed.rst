* Updated multi-arm keyboard reset IK to aim the waiting arm at its first upcoming
  sequence key in its own half, while the other handles the current key or
  Backspace. When no upcoming key belongs to the waiting arm, its target is
  sampled from its own half.
* Updated SO101 keyboard resets to start with the camera mount upright and the gripper
  partly open. Each arm's IK uses the left or right fingertip according to its
  target's position, including the moving finger's joint transform.
