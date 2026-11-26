NeIO transfer and convergence study
===================================

Purpose and scientific scope
----------------------------

Test the author's verbally communicated hypotheses: cloth-only training transfers
to other physics, and one/few learned sweeps can compete with much larger classical
solver budgets. The paper and original training code are unavailable. The supplied
port uses separate particle and rigid networks; our shared-vector network is an
explicit experimental extension. Report the supplied checkpoints separately.

The nine original toy pilots are cancelled and archived in W&B under
``superseded-toy-pilots``. They are not evidence for these hypotheses. Production
launches reject their configuration and any replacement containing diagnostic
fixtures. Small fixtures remain only in numerical and logging tests.

Workloads and data
------------------

Deformable scenes instantiate the pinned Newton examples directly. Their meshes,
controllers, articulated mechanisms, self-contact and native time integration
produce complete trajectories. No example methods, dependency files or global
solver factories are patched. The measured material multiplier modifies model
arrays through Newton's update interface. The study declares each duration and
material/motion range in ``study.json``.

Captured native examples execute one original warmup frame to allocate captured
collision buffers before material variation and recapture. That frame counts
toward the configured physical duration. Its allocator graph remains alive while
the scene uses its buffers; discarding it invalidates subsequent GPU execution.

* Cloth twist: 2,500 vertices, opposite edge rotation, tight folds and self-contact,
  12 seconds. Training varies rotation speed and stiffness. Test includes faster
  and reversed twisting.
* Cloth rollers: the original rolled cloth and two rotating rollers, 8 seconds.
  Preserve its centimetre simulation units; ``length_unit_m=0.01`` records the
  conversion for presentation and geometric thresholds.
* Cable plectoneme: 80 segments, 2-second settling, 8-second counter-twist and
  3-second hold. Require out-of-plane buckling and non-neighbour self-contact.
* Cable pile: 100 cables with 40 segments each, 6 seconds of falling, entangling
  and frictional contact.
* Hanging soft bodies: four tetrahedral beams with different damping, 6 seconds.
* Soft-body/cloth impact: a tetrahedral body falls onto a 40-by-40 cloth grid;
  coupled deformation and self-contact continue for 6 seconds.
* Dense rigid pile: 36 training / 54 test bodies, including the native bear mesh,
  cones, spheres, boxes, capsules and cylinders. Eight seconds of gravity and
  alternating horizontal acceleration produce sustained multi-body contact.
  Require at least 15 distinct dynamic body pairs and at least three simultaneous
  pairs in half the sampled frames. Floor contact alone cannot satisfy this gate.
* Cable cross-slide: a closed cable loop, passive pulleys and two articulated
  slides. Only two input pulleys are commanded. Run beyond a full 16-second
  rectangle, retaining the native tracking-error check.
  This is cable coverage, not a substitute for a robot articulation workload.
* Franka FR3: the actual imported arm, seven driven revolute joints and two
  prismatic fingers, a tool mass and a 30 N downward wrist load (60 N held out).
  Position targets drive dynamic links; link poses are not prescribed.
* ANYmal D: floating base, twelve driven leg joints, unequal foot support heights,
  and an oscillating lateral load of 80 N (160 N held out). Both robot workloads
  run for eight seconds and must exercise at least six revolute joints by 0.1 rad;
  the quadruped must also spend at least half its samples in ground contact.
  MJWarp generates both robot banks; require joint-anchor separation below 5 mm.
  NeIO and VBD still solve the same held-out states during evaluation. The initial
  VBD robot generator was rejected because its loaded arm separated joint anchors
  by about 10 cm; valid joint geometry cannot be inferred from finite coordinates.
* Loaded cloth bag: rigid shapes settle inside a deformable bag, which is lifted
  while supporting their weight, 7 seconds of two-way coupling.
* M12: authored NIST threaded nut and bolt collision meshes at 0.1 mm SDF voxel
  resolution. A bounded PI torque controller turns the nut in an aligned spindle
  that permits axial sliding and rotation; axial motion must emerge from thread
  contact. The initial nut is raised 10.5 mm from the authored assembly position
  so the stroke stays within the usable thread instead of hitting its runout.
  No helical joint or prescribed insertion is used. The separate static
  axial/lateral stress cases retain a free six-DOF nut. Reversing every three
  seconds, eight-second training / twelve-second test
  trajectories use 20 N preload and must accumulate at least four turns. The
  engagement gates require contact in 90% of samples and axial travel within
  2 mm of the 1.75 mm thread lead throughout the trajectory. Separate
  eight-second turning tests apply 200/400 N with a 2 N m torque limit, in addition
  to the axial/lateral static stress cases. Record net and accumulated turns,
  axial travel, travel per turn,
  penetration and escape; a completed trajectory is not automatically successful
  engagement. Test loads do not enter training.
  The motor integral belongs to the scene controller during trajectory generation
  and rollouts. Cold timestep probes retain the recorded torque/force command and
  do not advance that controller. Large explicit proportional gains were rejected
  because they caused discrete-time torque oscillation rather than threading.

Each native trajectory supplies approximately 96 sampled timestep problems;
the new rigid and robot workloads sample at least 30 Hz to capture impacts and
avoid rotation aliasing. Export unprojected recorded body poses alongside the
common IK/FK-projected probe states. Rigid replay geometry uses pose interpolation,
not linear interpolation of rotating world-space vertices. The cloth example's
kinematic particle rollers are also reconstructed as rigid transforms.

Samples cover early motion and late difficult interactions. Save positions, velocities,
external forces and controller arrays together. The native solver generates the
state distribution; it supplies no target solution or supervised label. Each sampled
state starts eight consecutive timesteps driven by the learned solver, with the
sampled external command held fixed. Intermediate positions and velocities come
from the model, so training includes its own accumulated errors. Optimizer memory
resets at each timestep in both training and evaluation; it persists across the
sweeps within that timestep. The recurrent step scale is bounded between 1e-10
and 1e-2 in the shared vector coordinates in both modes.

The experimental trainer minimizes ``log(1 + ||g_after||² / ||g_before||²)`` per
sweep and world, with a 1e-12 floor on the denominator. Rigid torques are divided
by each body's radius of gyration before computing the residual. Two additional
physics-force evaluations approximate the Hessian-vector product needed to
differentiate this loss. Collision topology, planar truncation coefficients,
history and preceding timesteps remain detached. This is an explicit residual
objective, not the author's unavailable training recipe or a scalar energy loss.
Training accounts for all four force evaluations per sweep; inference retains
one force evaluation per sweep, plus separately recorded diagnostics.

Cloth-only, deformables (cloth/cable/volume), and mixed (also rigid/articulated/
rigid-soft/threaded) arms share architecture, initialization per seed, update budget
and optimizer. Independent GPU processes sample complete scenes and average model
gradients. ``--worlds=1`` is required; the former tiny replicated worlds are gone.
Family rotation balances training; within each family it rotates workloads. Seeds
101/202/303 control initialization, material variation and sampled times. Validation
and test have separate seeds and declared material/motion ranges. Report any fixed
native benchmark with overlapping motion coverage explicitly; seed separation
alone does not make deterministic geometry a held-out distribution.

Ownership
---------

``run.py`` is the composition root: config, distributed processes, network, AdamW,
scene-bank cache, checkpoints, W&B and outputs. ``scenes.py`` constructs native
examples, dense-pile/robot geometry and their controls, and M12 geometry/forcing;
native examples own their prescribed motion.
``optimizer.py`` owns shared-vector encoding and the force-to-weight derivative.
``benchmark.py`` owns trajectory sampling, coverage checks, equal-state probes and
full native rollouts. ``study.json`` owns the experiment matrix. ``test_study.py``
protects the physical derivative, data/launch boundary, real folded-state learning
and experiment identity. No secondary launcher or trainable scene-local network.

The source snapshot under ``_source/`` is a pinned read-only dependency. Its
revision, package versions and file hashes are checked before use. The image
contains the snapshot, authored M12 assets, and pinned robot meshes/URDF/USD files;
they are excluded from Git and checked against the asset hash manifest.

Verification and evaluation
----------------------------

``--mode=verify`` runs complete workloads, checks finiteness and collision capacity,
records native task checks and the buckling/self-contact/tracking evidence, and
uploads state banks plus ``coverage.json`` to W&B. It does not train a model.
A missing event or collision overflow fails bank construction. A native task check
is recorded separately from finite-state validity; a solver failure is retained,
never silently omitted from comparisons.

Fast evaluation uses five common states spanning each complete trajectory. Sweep
NeIO at 1/2/4/8/16/32, VBD and MJWarp at 1/2/4/8/16/32/100. Record actual MJWarp
iterations, cold-start residual reduction, geometric error, and both eager and
captured substep time including collisions. A capture that disagrees with ordinary
execution has no valid captured timing; its eager measurements remain available
and its rollout uses eager execution. Contact matching is enabled for VBD history,
and diagnostic collision passes use separate storage during rollouts and state-bank
generation. Separately check references at 200/400
iterations. VBD-100 is a comparison, not ground truth. Long evaluation replaces
the native scene's solver instance and reuses its original controller over the
full configured duration, at a bounded subset of sweep budgets.

VBD compliant ALM, NeIO cubic penalty, and MJWarp constraints differ. Cross-solver
physical outcomes and timing are meaningful; their unlike residuals are not a
common energy. Native contact history and particle collision filtering are retained
for VBD; the supplied NeIO objective lacks hard contact history and some native
filter/scheduling options. These are formulation differences, not claimed solver
parity. MJWarp cannot represent the cable or particle objectives. The pinned adapter
does not expose PGS; report unsupported rows instead of calling Newton iterations
PGS sweeps.

Tracking and launching
-----------------------

Online W&B is required by default; initialization failure stops all ranks. The
project defaults to ``neio-transfer`` and the authenticated entity. The primary
curves measure before/after objective-gradient norm ratios on five fixed validation
states, starting at update zero, plus worst geometric errors and failure fraction.
A ratio below one means the solver reduced the residual; it does not prove minimum
energy. ``train/loss`` is the explicit normalized residual objective above; about
0.693 means no residual improvement on a sweep. Training residual ratios are separated by
workload and GPU, because raw magnitudes from different physics cannot be compared.
Evaluation publishes physical error, reference distance, failure and timing curves
against sweep count directly to W&B, alongside the detailed JSON artifact.
Single-timestep physical passes and full-rollout failures have separate curves.
Captured timing is omitted for an entire comparison if any probe fails capture parity.

Workload names, sample times and accounting remain in the raw logs. Every 32 updates save an atomic
checkpoint; at validation (256 updates) and completion upload latest/best model
artifacts with provenance and logs. The untrained baseline participates in best
checkpoint selection, and evaluation loads the best checkpoint and records its
training step. Checkpoints preserve W&B identity for resume.
Full training banks and evaluation reports are also artifacts. Offline mode is
explicit and intended for local tests.

Use the existing toolkit and always overwrite the sole user-facing tag ``lab``::

    uv run --project scripts/neio scripts/neio/run.py --mode=verify --run_name=native-verification
    uv run --project scripts/neio scripts/neio/run.py --mode=preflight
    ./buildnpush.sh lab --source
    ./submit.sh scripts/neio/run.py image=lab kitless=1 python_project=scripts/neio num_gpu=2 --mode=train --worlds=1 --arm=cloth,deformables,mixed --seed=101,202,303 --eval-after

Record the immutable image digest and Git revision in each submission. Isaac Lab
and this study use separate locked environments because the source port requires
MuJoCo/MJWarp 3.12. ``python_project=scripts/neio`` selects the study environment.
OSMO retains outputs beneath ``ISAACLAB_OUTPUT_DIR``; local defaults use
``logs/neio/<run_name>``. Resume is explicit and preserves configuration and RNG.
