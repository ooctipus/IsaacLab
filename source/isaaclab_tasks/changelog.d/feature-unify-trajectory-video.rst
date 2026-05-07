Added
^^^^^

* Added :class:`~isaaclab_tasks.manager_based.multi_task.terrain.mdp.record_trajectory_video` event term and registered it on the position-locomotion env, producing a top-down 2D trajectory video that overlays a 10% subsample of robot dots on the same terrain heightmap as the curriculum spawn-scatter.
* Added :class:`~isaaclab_tasks.manager_based.multi_task.terrain.viz.TrajectoryRecorder` -- a numpy + Pillow render path (no matplotlib draw cycle) with shared-master-palette gif quantization, sized for ~1.6k dots × 200 frames at ~1.4 s and ~10 MiB.
* Added :func:`~isaaclab_tasks.manager_based.multi_task.terrain.viz.render_terrain_background` and :func:`~isaaclab_tasks.manager_based.multi_task.terrain.viz.heightmap_to_rgb`, the shared raycast → heightmap → RGB utility now used by both the spawn-scatter curriculum panel and the trajectory video recorder.
