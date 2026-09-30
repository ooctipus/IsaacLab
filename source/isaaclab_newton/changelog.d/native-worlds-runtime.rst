Added
^^^^^

* Added experimental ``NewtonWorldsBackend`` and ``NewtonWorldsCfg`` for context-owned
  native world populations with GPU lifecycle commands, shared virtual backing,
  and reset-only/current-step pose refresh. Native state remains authoritative in
  ``MuJoCoWorlds``; prepared task callbacks supply controls, resets and contacts.
