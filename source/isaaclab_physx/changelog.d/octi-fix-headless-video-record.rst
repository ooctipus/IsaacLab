Fixed
^^^^^

* Fixed :func:`~isaaclab_physx.renderers.kit_viewport_utils.set_kit_renderer_camera_view`
  authoring only the camera xform via USD, which left no RTX render context bound to
  ``/OmniverseKit_Persp`` under the headless rendering Kit experience and caused
  ``--video`` captures to be all-zero. The helper now routes through
  :class:`isaacsim.core.rendering_manager.ViewportManager`, which both creates the
  camera prim (if absent) and wires it into the RTX render path.
