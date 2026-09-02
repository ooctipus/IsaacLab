Changed
^^^^^^^

* Integrated ``OvPhysxManager`` with the single cfg-owned clone lifecycle. Declarative and
  homogeneous direct scenes now provide its published plan; low-level tools may continue to call
  ``OvPhysxReplicateContext(sim).replicate(plan)`` directly.
