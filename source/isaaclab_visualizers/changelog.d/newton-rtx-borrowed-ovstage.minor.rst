Added
^^^^^

* Added :class:`~isaaclab_visualizers.newton.NewtonRTXStageVisualizer` and its
  :class:`~isaaclab_visualizers.newton.NewtonRTXStageVisualizerCfg` (``newton_rtx_stage``), which render the
  simulation's own cloned USD stage through Newton's ``ViewerRTX``, so MDL materials such as glass are
  visible. It requires a Newton release that accepts ``ViewerRTX(ovstage=...)`` and OVRTX 0.5 with OVStage 0.2.
  The existing ``newton_rtx`` camera presenter is unchanged.
