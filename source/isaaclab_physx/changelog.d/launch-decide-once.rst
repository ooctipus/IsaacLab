Changed
^^^^^^^

* :class:`~isaaclab_physx.app.KitLauncher` reads whether the run has the Kit visualizer from the ``visualizer``
  selection (``--visualizer kit``) instead of the removed ``kit_visualizer`` launcher argument. It auto-starts XR,
  and runs headless, only when ``--visualizer`` does not select ``kit``.
