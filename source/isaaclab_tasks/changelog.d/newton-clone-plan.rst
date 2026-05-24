Changed
^^^^^^^

* Changed :class:`~isaaclab_tasks.direct.franka_cabinet.FrankaCabinetEnv` to
  resolve robot link frames through the active
  :class:`~isaaclab.cloner.ClonePlan` instead of hardcoded ``/World/envs/env_0`` paths.
