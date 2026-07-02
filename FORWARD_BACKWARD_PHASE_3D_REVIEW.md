# Forward/Backward Phase 3D Historical Review

Status: superseded historical checkpoint; Phase 3 is complete.

This file records a retired Phase 3D midpoint and is not an active
implementation or evidence contract. Its `MotionRobotPreset`,
`SmplHumEnvMaterializer`, `G1LafanMaterializer`,
`materialize_motion_bank`, packed-bank layout hashes, materializer
benchmark, 55-test gate, and Phase 3E handoff no longer name the current
architecture.

The completed architecture uses:

- `scene.robot` as the sole physical robot authority;
- command-owned `MotionTaskTable` named columns and sampling law;
- live-articulation-aware `SmplHumEnvFrameBuilder`,
  `G1LafanFrameBuilder`, and `G1SmplHumEnvFrameBuilder`;
- one `MotionImitationEnvCfg` resolved through normal preset axes;
- one unified RSL-RL learner path.

Current task-table evidence is frozen by:

- G1-LAFAN v4 receipt:
  `0d639e5800ac1309ab2a697d66e69a9feca4da4bc8c8ce0b05a2365a39beae8d`;
- SMPL-CMU v4 receipt:
  `ea020946950e8e96adbc744830a785fc27895da023e4948c71dd0b987a030eb3`.

The authoritative completed design, execution record, and Phase 4 stop
boundary are in
[FORWARD_BACKWARD_PHASE_3_ENVIRONMENT_PLAN.html](FORWARD_BACKWARD_PHASE_3_ENVIRONMENT_PLAN.html#execution).

Final Phase 3 verification: **308 passed**. All Phase 3 scoped hooks
passed. Repository-wide hooks were also run and exposed only unrelated
pre-existing branch failures, so they are not an open Phase 3 gate.
