* Fixed the in-hand reorientation tasks' reset starting the cube inside the
  hand's fingers: the hand's joints are now reset first and the cube's drawn
  pose is lifted against gravity by the least distance that clears every hand
  collider (:class:`~isaaclab_tasks.core.reorient.clearance.ObjectClearance`).
