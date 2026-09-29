Added
^^^^^

* Added :func:`~isaaclab.visualizers.visualizer_cfg.parse_visualizer_csv` and
  :func:`~isaaclab.visualizers.visualizer_cfg.resolve_visualizer_cfgs`, which parse a ``--visualizer`` selection
  and apply it to a list of visualizer configs, and
  :data:`~isaaclab.visualizers.visualizer_cfg.VISUALIZER_TYPES`, which maps each visualizer type to its default
  config class.

Changed
^^^^^^^

* **Breaking:** Tasks, ``run_cartpole_rl_env.py``, ``lift_franka_soft.py`` and ``check_keyboard.py`` no longer
  open a visualizer by default. Pass ``--visualizer`` (for example ``--visualizer kit`` or
  ``--visualizer newton_gl``) to open one. Demos, examples and visualizer tutorials keep their default visualizer
  through ``parser.set_defaults(visualizer=[...])``; ``--visualizer`` replaces it.
* **Breaking:** Without ``--visualizer``, :func:`~isaaclab.app.launch_simulation` runs no visualizer, even if
  :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` lists some. ``--visualizer`` selects which visualizers
  run and the configured ones only supply the settings of the selected types: each selected type uses the
  configured visualizer of that type, or else its default config. Configured visualizers no longer start Kit,
  OVRTX, or cameras unless selected. To keep running a visualizer your config lists, pass
  ``--visualizer <type>`` (or ``launch_simulation(cfg, {"visualizer": "<type>"})``). A
  :class:`~isaaclab.sim.SimulationContext` built without a launch still runs the configured visualizers as given.
* :func:`~isaaclab.app.launch_simulation` decides the run's visualizers and device once and writes them to the
  :class:`~isaaclab.sim.SimulationCfg` of the launched config: ``visualizer_cfgs`` holds exactly the visualizers
  that run (``--visualizer`` and ``--max_visible_envs`` applied) and ``device`` holds the resolved device
  (``--device``, the per-rank GPU when distributed, the runtime's refinement, with ``cuda`` pinned to an index).
  A launch without ``--device`` starts the runtime on the config's device.
* :class:`~isaaclab.sim.SimulationContext` creates the visualizers of
  :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` as given, and
  :meth:`~isaaclab.sim.SimulationContext.has_active_visualizers` counts configured non-headless visualizers.
* :class:`~isaaclab.sim.SimulationContext` normalizes :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` to a
  list, so a single config becomes a one-element list and None becomes ``[]``.

Removed
^^^^^^^

* **Breaking:** Removed the ``/isaaclab/visualizer/explicit`` and ``/isaaclab/visualizer/disable_all`` settings,
  and ``/isaaclab/visualizer/types`` and ``/isaaclab/visualizer/max_visible_envs`` hold the launch's selection
  (``types`` is a comma-separated list, empty when none was selected).
  Read :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` or
  :meth:`~isaaclab.sim.SimulationContext.resolve_visualizer_types` instead.
* **Breaking:** Removed the ``visualizers`` argument of :func:`~isaaclab.sim.build_simulation_context`, which only
  set a setting and never created the visualizers. Set :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` on
  the ``sim_cfg`` you pass instead.
* **Breaking:** Removed ``--visualizer none``. Omit ``--visualizer`` to run without visualizers.
* **Breaking:** Removed the ``visualizer_intent`` launcher argument of :func:`~isaaclab.app.launch_simulation`
  and the ``kit_visualizer`` launcher argument it wrote. Pass ``visualizer="kit"`` to request the Kit
  visualizer.
