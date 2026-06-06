Fixed
^^^^^

* Fixed black-frame output from ``--video`` in headless rendering. The
  ``apps/isaaclab.python.headless.rendering.kit`` experience was missing the
  ``isaacsim.core.rendering_manager`` dependency, so
  :func:`omni.replicator.core.create.render_product` attached to a USD camera with
  no RTX render context and every captured frame was zero-filled.
* Fixed spurious ``AttributeError: 'NoneType' object has no attribute 'shared_metatype'``
  raised from :meth:`~isaaclab.assets.Articulation.body_names` during environment teardown
  when ``ViewerCfg.origin_type`` is ``"asset_root"`` or ``"asset_body"``.
  :meth:`~isaaclab.envs.ManagerBasedEnv.close` (and the analogous methods on
  :class:`~isaaclab.envs.DirectRLEnv` / :class:`~isaaclab.envs.DirectMARLEnv`) now tears
  down :class:`~isaaclab.envs.ui.ViewportCameraController` before calling
  :meth:`~isaaclab.sim.SimulationContext.stop`, so the Kit app-update pump inside
  ``stop()`` no longer fires the controller's post-update callback against assets whose
  physics views were just invalidated.
