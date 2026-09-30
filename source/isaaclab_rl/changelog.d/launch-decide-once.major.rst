Changed
^^^^^^^

* **Breaking:** The zero and random agents no longer open the Newton GL visualizer by default. Pass
  ``--visualizer newton_gl`` to open it.
* The run summary of the reinforcement learning workflows is printed after the launch and reports the resolved
  backends, visualizers, and device, so a run without visualizers shows ``headless`` and distributed runs
  show each rank's device. Automatic backend selectors are no longer shown next to the backend they resolved to.
* **Breaking:** ``--video`` takes an optional source, ``--video [SOURCE]``, and ``--visualizer`` still decides
  which visualizers open a window:

  * ``--video`` (``--video viz``): the first capture-capable visualizer ``--visualizer`` selects, else a headless
    ``newton_gl``, also when only streaming visualizers such as ``viser`` or ``rerun`` are selected.
  * ``--video viz:<type>`` (``kit``, ``newton_gl``, ``newton_rtx``): the selected visualizer of that type, else an
    extra headless one, e.g. ``--video viz:newton_gl --visualizer viser``.
  * ``--video sensor:<name>[:<channel>]``: that scene sensor; no visualizer is added.

  A Hydra override is never taken as the source: ``--video presets=newton_mjwarp`` records from ``viz`` and
  applies the preset. ``--video`` without ``--visualizer`` now records from a headless ``newton_gl`` instead of a
  headless Kit; pass ``--video viz:kit`` for the previous behavior. Replace ``visualizer:<type>`` sources with
  ``viz:<type>``.
* **Breaking:** :func:`~isaaclab_rl.entrypoints.common.pre_launch_video_config`, called before
  :func:`~isaaclab.app.launch_simulation`, adds a recorder for the ``--video`` source unless the environment config
  declares recorders, and no longer selects ``--visualizer kit`` or sets ``headless``; the launch resolves the
  source. :func:`~isaaclab_rl.entrypoints.common.apply_video_recording`, called inside the launch, only applies the
  output directory, ``--video_length`` and ``--video_interval``. The zero and random agents now configure
  recording after the launch.
* :func:`~isaaclab_rl.entrypoints.common.enable_cameras_for_video` only enables cameras for
  ``--capture_env_sensors``; the launch enables the rendering a video source needs.
* The ``video`` field of the :mod:`isaaclab_rl.entrypoints.api` requests takes a ``--video`` source string as
  well as a bool.

Removed
^^^^^^^

* **Breaking:** Removed the ``apply_device`` argument of :func:`~isaaclab_rl.entrypoints.common.apply_env_overrides`,
  which no longer writes ``--device``; :func:`~isaaclab.app.launch_simulation` writes it to
  ``env_cfg.sim.device``. Call :func:`~isaaclab_rl.entrypoints.common.show_run_summary` inside
  :func:`~isaaclab.app.launch_simulation`.
* **Breaking:** Removed :func:`~isaaclab_rl.entrypoints.common.validate_distributed_device`: with ``--distributed``,
  :func:`~isaaclab.app.launch_simulation` always assigns each rank a ``cuda:N`` device, so the check could not fail.
