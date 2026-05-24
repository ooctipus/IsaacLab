Changed
^^^^^^^

* Changed :class:`~isaaclab_ov.renderers.OVRTXRenderer` to derive camera paths
  and scene-partition tokens from
  :func:`~isaaclab.cloner.cloner_utils.expand_clone_plan_paths` instead of
  hardcoded ``/World/envs/env_*`` formulas.
