Removed
^^^^^^^

* **Breaking:** Removed runtime PPISP stage discovery, ``PpispCfg.camera_prim_path``, and
  the auto-discovery helpers. Import source-camera attributes with
  ``ppisp_cfg_from_usd_camera`` and pass the resulting explicit ``PpispCfg`` to the camera
  configuration before constructing the scene.
