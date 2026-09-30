Recording Animations of Simulations
===================================

.. currentmodule:: isaaclab

Isaac Lab records animations with the **OVD Recorder**. It uses OmniPVD to capture simulated physics from a played stage
and then **bakes** that directly into an animated USD file. It runs through CLI arguments without requiring dynamic
physics state to be mirrored into USD.
The animated USD can be quickly replayed and reviewed by scrubbing through the timeline window, without simulating expensive physics operations.

.. note::

  Omniverse only supports **either** physics simulation **or** animation playback on a USD prim—never both at once.
  Disable physics on the prims you want to animate.


OVD Recorder
------------

The OVD Recorder uses OmniPVD to record simulation data and bake it directly into a new USD stage.
This method is more scalable and better suited for large-scale training scenarios (e.g. multi-env RL).

It’s not UI-controlled—the whole process is enabled through CLI flags and runs automatically.

.. note::

   The OVD Recorder uses OmniPVD, which only records **PhysX** simulations. If the active physics backend is not
   PhysX (for example, Newton, which is the default for many tasks), Isaac Lab raises an error at startup naming
   the active backend instead of recording. Select the PhysX backend by adding ``physics=isaacsim_physx`` to the
   command line, as shown below.

   The PhysX backend requires Isaac Sim. If it isn't installed yet, add ``--extra isaacsim`` to the ``uv run``
   command; see :ref:`installation-optional-extras` for details.


Workflow Summary
~~~~~~~~~~~~~~~~

1. User runs Isaac Lab with animation recording enabled via CLI
2. Isaac Lab starts simulation
3. OVD data is recorded as the simulation runs
4. At the specified stop time, the simulation is baked into an outputted USD file, and IsaacLab is closed
5. The final result is a fully baked, self-contained USD animation

Example Usage
~~~~~~~~~~~~~

To record an animation:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         uv run python scripts/tutorials/03_envs/run_cartpole_rl_env.py --anim_recording_enabled --anim_recording_start_time 1 --anim_recording_stop_time 3 physics=isaacsim_physx

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         uv run python scripts\tutorials\03_envs\run_cartpole_rl_env.py --anim_recording_enabled --anim_recording_start_time 1 --anim_recording_stop_time 3 physics=isaacsim_physx

.. note::

   The provided ``--anim_recording_stop_time`` should be greater than the simulation time.

.. warning::

   Currently, the final recording step can output many warning logs from [omni.usd]. This is a known issue, and these warning messages can be ignored.

After the stop time is reached, a file will be saved to:

.. code-block:: none

  anim_recordings/<timestamp>/baked_animation_recording.usda


.. _Omniverse Launcher: https://docs.omniverse.nvidia.com/launcher/latest/index.html
