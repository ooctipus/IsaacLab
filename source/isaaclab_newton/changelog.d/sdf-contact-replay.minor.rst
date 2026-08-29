Added
^^^^^

* Added
  :attr:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg.sdf_contact_replay_max_per_world`
  to size Newton's bounded cache per replicated world for exact replay of
  unchanged contacts between sleeping dynamic and kinematic SDF shapes. Set it
  to a positive row count with solver sleeping enabled; zero keeps replay
  disabled.

Changed
^^^^^^^

* Preserved box geometry when strict all-shapes SDF provisioning was enabled.
  Existing strict all-shapes SDF configurations require no changes.

Fixed
^^^^^

* Fixed selective Newton and MJWarp world resets to invalidate cached collision
  history for the reset worlds.
