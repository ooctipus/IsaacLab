NeIO transfer and convergence study
===================================

Status and hypotheses
---------------------

The paper and original trainer are not available. The cloth-only transfer and
one/few-sweep performance claims were communicated verbally to the user. This
study tests those hypotheses; it is not an exact reproduction of the unpublished
paper. The supplied port has separate 3D particle and 6D rigid networks and an
inference-only solver. Its rigid checkpoints were trained on rigid scenes.

Two results must remain distinguishable:

* The supplied checkpoints, evaluated with their original architecture.
* Our shared-vector training experiment, where exactly the same learned weights
  propose particle, rigid translation, and rigid rotation updates. This mapping
  is an experimental extension, not an attributed feature of the paper.

Training arms
-------------

All arms start from the same random initialization for each seed. They have the
same model, optimizer, number of parameter updates, and number of sampled worlds.
Also record oracle calls, processed nodes, and elapsed GPU time: equal updates
do not imply equal compute when the scene types have different costs.

1. Cloth only: randomized drape, stretch, bend, contact, and moving pins.
2. Deformables: cloth plus cable and volumetric soft-body scenes.
3. Mixed: the above plus rigid contacts, articulated chains, and two-way coupling.

Use independent train, validation, and test seeds and parameter ranges. Validation
monitors progress; the primary comparison uses the final checkpoint after the
same update budget. The test suite never selects weights or hyperparameters.
Severe 200/400 N M12 loads and the most violent cable/cloth motions are held out.
Run a short pilot before allocating longer runs. Use three training seeds for
final comparisons. Report every run, including numerical failures.

Ownership and file tree
-----------------------

``run.py`` is the composition root and owns CLI configuration, distributed rank
setup, random seeds, the shared network, AdamW, checkpoints, and output paths.
Each rank owns one Newton process and its scene states. Ranks draw scene families
from the arm's mixture and average parameter gradients before the update. No
scene owns a separate trainable copy with independent optimizer state.

``scenes.py`` owns physical models, parameter sampling, initial states, prescribed
motion and applied forces. It knows nothing about network weights or training.

``optimizer.py`` owns the shared-vector mapping, local frames, feature/history
normalization, and the training bridge from post-update forces to weight
gradients. Newton remains the owner of the physics and collision evaluation.

``benchmark.py`` owns fixed-state convergence probes and rollout measurements.
It consumes scene/model/checkpoint instances passed by the root. It does not
construct a second training policy or silently change physics parameters.

``study.json`` owns the ablation and evaluation matrix. ``source.lock.json``
identifies the supplied source revision. ``test_study.py`` protects gradient
sign/frame transformations, parameter sharing, dataset separation, and reporting
of failures. The benchmark selects its solver instances directly; no registry, compatibility
aliases, or secondary launchers are introduced for this study.

The supplied Newton/NeIO code is a pinned, read-only dependency snapshot under
``_source/``, excluded from Git and included in the container build. It is not a
copy of the user's modified working tree. The snapshot's license and provenance
travel with the image. Research code must not patch that snapshot at runtime.

Learning signal
---------------

Each iteration evaluates the current timestep objective gradient through Newton,
forms the network inputs, proposes an update, applies it, then evaluates forces
again. The post-update forces provide the derivative for learning the update.
VBD does not supply a target position. Inputs, collision topology, fitted frames,
and carried history are detached for a first-order, one-iteration training graph.
This truncation is a declared training choice, not full differentiation through
the simulation or an assertion about the author's trainer.

Use a single 3-vector interface. A particle contributes one vector node; a rigid
body contributes translation and rotation vector nodes. A characteristic body
length converts angular displacement to length and torque to force, preserving
virtual work. The same encoder, message-passing block and output heads serve
every vector node. Cloth-only training therefore leaves no untrained rigid head.
This representation itself is an experimental factor; do not attribute its
transfer results to the original two-network architecture.

Train through the applied displacement, including the declared treatment of
collision truncation. Verify the force-to-weight gradient numerically before
launching training. Do not label a gradient dot-product surrogate as a measured
energy. Report gradient residuals and physical errors when a matching scalar
energy evaluator is unavailable.

Evaluation
----------

Use a fast fixed-state suite and a smaller long-rollout suite. Rendering is off
for timing; compilation, initialization, and asset loading are measured separately.
Each solver receives the same saved physical state, timestep, forcing, collision
settings and material values where its formulation supports them. Reset solver
history for cold-start probes; test carried history in a separately named arm.

Measure NeIO at 1, 2, 4, 8, 16, 32 sweeps, VBD at 1, 2, 4, 8, 16, 32, 100, and
rigid MJWarp Newton/PGS at 1, 2, 4, 8, 16, 32, 100. Record actual MJWarp iterations,
not only the configured cap. A sweep is one complete solver update; timings
include its gradient computation, network inference, and required collision work.
Also report end-to-end substep time and collision frequency. PGS is not claimed
to support the particle cloth/cable/soft-body cases it cannot represent.

Cover rigid stacking and impact, mass ratios, frictional sliding/jamming,
articulated chains and drives, cloth drape/twist/self-contact, cable bending,
whipping and twisting, volumetric deformation, and two-way rigid-soft contact.
The M12 test must use actual threaded collision geometry, not cylinders or a
helical joint that removes the difficult contact problem. Sweep axial and lateral
loads with explicit force application points, ramps, duration, and timestep.

Metrics: per-family success/failure rates, residual reduction versus sweeps and
milliseconds, penetration, joint gap, stretch, rest drift, and rollout divergence.
Use a high-budget reference only after checking its own convergence; VBD 100 is
a comparison arm, not automatic ground truth. Recompute contacts for geometric
penetration diagnostics. Preserve failed and unsupported cases in the report.

The supplied NeIO port uses cubic penalty contacts and plain penalty joints.
VBD and MJWarp have different formulations. Compare each solver's convergence
against its own objective, and compare physical outcomes across formulations.
Do not call a lower residual in a softer/different objective a solver speedup.

Launch contract
---------------

Build and overwrite only ``lab`` using the branch's existing ``buildnpush.sh`` /
``tools/buildnpush.py`` entry point. Submit through ``submit.sh`` with
``image=lab`` and ``kitless=1``. Never introduce a second image name for a retry.
Record the immutable image digest, source revisions, configuration, device, seed,
and workflow ID in every result. Checkpoint atomically, retain latest and best,
and support restart after preemption. A failed preflight must exit nonzero before
spending a training allocation.

Implemented protocol and current limits
---------------------------------------

The study has a separate locked Python project (``pyproject.toml``, ``uv.lock``).
The pinned NeIO Newton adapter needs MuJoCo/MJWarp 3.12; Isaac Lab's current
project needs 3.17. The existing cluster launcher selects exactly one environment
with ``python_project=scripts/neio``. Both environments live in the ``lab`` image.
The study refuses mismatched package versions at startup. ``source.manifest.json``
checks the dependency and asset bytes before running.

Training resets learned history at each timestep (four iterations by default),
uses a detached 1e-3 N floor for objective-gradient normalization, and clips the
parameter-gradient norm to one. Carrying an untrained step-scale history across
timesteps collapsed the step size in a pilot. Carried history is still evaluated
separately. A checkpoint is written every 32 steps, at an episode boundary;
validation runs every 256 steps and at the end. Only cloth validation contributes
to selection of the auxiliary best checkpoint, so rigid/cable validation cannot
select the cloth-only model. Automatic post-training evaluation uses the final
``latest.pt`` checkpoint to preserve equal update budgets. ``latest.pt`` includes optimizer,
per-rank random state and the best weights so a restart is self-contained.
``best.pt`` is for evaluation. Restart may repeat at most 31 uncheckpointed steps.
The default 9984 updates give 52 complete six-family rounds on one rank.

VBD explicitly uses compliant ALM (the recommended current formulation) and
creates probe trajectories. The legacy default diverged in the randomized stack
preflight and is not used as the headline VBD baseline. Before saving a rigid/articulated probe, Newton
IK/FK projects it to consistent joint and body coordinates, and every solver
receives that same projected state. Without this projection, MJWarp would read
stale joint coordinates from a VBD trajectory. Full evaluations record distances
to VBD-400 and MJWarp-400 alongside each reference's change from 200 to 400.
These are budget-stability checks, not certificates of optimality. Distances
across formulations are labeled explicitly.

Full probes report both eager and captured CUDA timing, excluding warmup and
capture construction. Captured and eager final states must agree. The diagnostic
force pass is outside timing. Quick probes omit capture and high-budget reference
checks. Full long rollouts run at NeIO 1/4 sweeps and VBD/MJWarp 8/100;
all configured sweep budgets still receive fixed-state probes. Quick mode runs
its single configured budget through both probes and rollouts. Each row records
``rollout_steps`` (zero means no rollout was tested), so a probe pass must not be
interpreted as long-term stability. Particle residuals have force units; the port's body residual combines
force and torque numerically and is labeled ``mixed_units``. It is only a
within-state relative diagnostic, not an energy or a cross-object physical norm.

PGS rows are recorded as unsupported: this Newton MJWarp adapter exposes only
Newton and CG. VBD allocates 1024 contacts per body for the threaded geometry;
its usual 64-contact default drops M12 contacts. NeIO's own force pass visits the
full global contact list rather than VBD's per-body list. Its inherited unused
list may still print a capacity warning. Global contact overflow invalidates a
measurement. Particle self-contact buffers are raised to 128 vertex / 256 edge
candidates; denser folds need a separate capacity study.

The test set includes 200/400 N axial and lateral M12 loads, opposite-edge cloth
twist, cable whip plus twist, stiff materials, mass ratios, driven articulation,
soft impact, coupling, and zero-gravity rest tests. Force ramps last 50 ms and act
at the nut's center of mass; M12 probes include 0, 60 and 120 ms. M12 uses 0.1 mm
SDF voxels and the authored threaded meshes. Mesh resolution, timestep refinement,
friction and contact-law calibration are further experimental factors: this
configuration alone cannot establish a universal speedup. Rendering is disabled.

``study.json`` declares geometric acceptance thresholds. A completed simulation
is distinct from passing those thresholds. The report includes penetration,
strain, joint gap, rest drift and nut escape displacement; rollout extrema are
sampled every 16 steps. Raw values remain available even when a threshold fails.
Those thresholds are research choices, not values attributed to the author.

Local use and submission
-------------------------

The dependency snapshot is intentionally absent from Git. Obtain the locked
revision of the source repository, then use ``git archive`` to extract ``newton``,
``experiments/neio``, ``experiments/neio-reference`` and ``LICENSE.md`` into
``scripts/neio/_source``. Download the two USD assets named in ``source.lock.json``
and export ``Usd.Stage.Open(path).Flatten()`` to the named ``assets/*.usda`` files.
The recorded hashes must match; do not silently replace missing dependencies.
The built image contains both the dependency and assets.

From the checkout root::

    uv run --project scripts/neio scripts/neio/run.py --mode=preflight
    uv run --project scripts/neio scripts/neio/run.py --mode=train --arm=cloth --steps=192 --worlds=2 --run_name=cloth-pilot
    uv run --project scripts/neio scripts/neio/run.py --mode=eval --checkpoint=logs/neio/cloth-pilot/best.pt --quick

Build and submit using the existing toolkit::

    ./buildnpush.sh lab --kitless
    ./submit.sh scripts/neio/run.py image=lab kitless=1 python_project=scripts/neio num_gpu=1 --mode=preflight
    ./submit.sh scripts/neio/run.py image=lab kitless=1 python_project=scripts/neio num_gpu=2 --mode=train --arm=cloth,deformables,mixed --steps=192 --worlds=4 --eval-after --quick --rollout-steps=256

Pass ``--image-digest`` and ``--git-revision`` with the actual published values
for recorded cluster runs. ``submit.sh`` expands comma-separated arguments into
sweeps, so use its bracketed-value syntax or the script defaults when an argument
itself contains commas. Three final training seeds are 101, 202, 303. The full
matrix should follow successful pilots; a quick pilot is not evidence for the
one-sweep performance claim.

OSMO supplies ``ISAACLAB_OUTPUT_DIR``; the study writes checkpoints, provenance,
training JSONL, validation JSON and evaluation JSON beneath that retained output
folder. Local runs default to ``logs/neio``. A run with an existing checkpoint
requires an explicit resume or a new run name.
