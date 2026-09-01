Changed
^^^^^^^

* **Breaking:** Made experimental manager-based environments construct declarative scene cfgs through
  the scene-owned clone lifecycle. Homogeneous direct environments now keep asset cfgs on the direct
  env cfg. Their ``_setup_scene()`` publishes the cfg-derived plan before constructing the prototype,
  then dispatches that same plan. Custom environment roots must supply all clone-owned cfgs before
  constructing assets.
* **Breaking:** Removed ``InteractiveSceneWarp``. Core ``InteractiveScene.reset()`` now accepts a
  Warp environment mask, so ``WarpFrontend`` preserves the scene class selected by the task cfg.
  Import ``InteractiveScene`` from ``isaaclab.scene`` for custom Warp scenes.
