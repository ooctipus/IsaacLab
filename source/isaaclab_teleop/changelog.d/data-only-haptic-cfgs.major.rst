Changed
^^^^^^^

* **Breaking:** Changed haptic feedback configs to select their runtime implementation with
  ``class_type``. Construct the implementation with ``cfg.class_type(cfg)`` instead of calling
  ``cfg.make_signal_fn()`` or ``cfg.build_sink(...)``.
* **Breaking:** Made :class:`~isaaclab_teleop.XrCfg` a scene-owned asset configuration. Declare it
  as ``scene.xr_anchor`` so the clone plan authors the anchor, remove ``IsaacTeleopCfg.xr_cfg``, and
  pass the same ``env_cfg.scene.xr_anchor`` explicitly to
  :class:`~isaaclab_teleop.IsaacTeleopDevice` or
  :func:`~isaaclab_teleop.create_isaac_teleop_device`.
* **Breaking:** Removed the deprecated ``OpenXRDevice``, ``ManusVive``, and their configuration
  classes. Use :class:`~isaaclab_teleop.IsaacTeleopDevice` with the plan-owned
  ``env_cfg.scene.xr_anchor``.
* **Breaking:** Removed ``XrCameraFeedCfg.enable_dlss_ray_reconstruction``,
  ``XrCameraFeedCfg.dlss_exec_mode``, and
  ``XrCameraFeedSession.requires_responsive_denoising``. Configure renderer policy on the camera's
  renderer config; XR picture-in-picture now uploads the camera's public RGBA buffer directly.
