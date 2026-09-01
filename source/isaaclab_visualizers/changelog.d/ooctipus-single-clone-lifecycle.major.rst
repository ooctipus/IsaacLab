Changed
^^^^^^^

* **Breaking:** Made ``KitVisualizerCfg`` request the shared USD scene clone context so Kit consumes
  topology from the active clone plan. Custom Kit visualizer configs should preserve
  ``cloning_contexts=("isaaclab.cloner:UsdReplicateContext",)`` instead of copying or exporting the
  completed USD stage separately.
