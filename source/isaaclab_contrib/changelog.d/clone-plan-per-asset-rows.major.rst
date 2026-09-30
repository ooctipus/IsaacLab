Changed
^^^^^^^

* **Breaking:** Moved the Newton deformable object implementation to
  :mod:`isaaclab_newton.assets`. Import ``DeformableObject`` and ``DeformableObjectData`` there;
  :mod:`isaaclab_contrib` now contains only optional coupling behavior and no Newton backend asset.
* **Breaking:** Changed ``VisuoTactileSensorCfg.prim_path`` to name the elastomer rigid body
  directly and made ``camera_cfg=None`` the only way to disable tactile camera rendering. The
  nested camera is clone-planned and constructed with the sensor before replication, then
  initialized through the shared sensor lifecycle.
