# Large-gain FPGS architecture study, 2026-09-11

This is the continuation of [the earlier architecture report](ARCHITECTURE_UPDATE_20260911.md)
and the inherited handoff, not a replacement benchmark baseline. The requested
work window is 10:37:35–20:37:35 UTC. No additional 2–4× whole-physics
improvement has been accepted. The useful outcome is a much cheaper isolated
proposal primitive and a source-backed row-free architecture to test next,
not a completed integrated solver improvement.

## Scope and comparison contract

- Change Newton FPGS/collision only. Isaac Lab runtime, harness, scheduling,
  timestep, substeps and effective iteration budgets remain unchanged.
- Prefer whole-stage elimination and different algorithm/data ownership over
  launch-parameter or percent-level tuning. Close architectures that fail their
  cost or physical-quality gate; do not optimize an already closed mapping.
- Numerical feasibility, complementarity, Coulomb dissipation, momentum
  roundoff and stability are the quality criteria. Matching a previous finite
  Gauss–Seidel iterate bit-for-bit is not the definition of correct physics.
- RTX PRO 6000 is primary; measure GB300 in parallel in independent processes.
  Component timings, work counts and optimistic scope budgets are not whole-step
  speedups. Do not credit unmeasured allocation, fallback or publication work.
- Keep the original worktrees and blocked Isaac Lab Newton pin unchanged.
  Newton work derives from the `ooctipus/newton` fork. These new handoff
  branches are documentation-only; private research source is archived locally,
  not installed as an enabled optimization by their commits.

The source baseline is Newton `a7eb7d15589a3c6032288a488463eb8e028d88e6`,
with original Lab runtime `1d8feb82d17dbfab8f0772de56f84deae2cb7974`.
The documentation branch derives from Lab `dfbf336e6d7d321db7d910c1ac58a7aba45cc9e8`.
Both original handoff commits remain ancestors. Experimental sources must not
become a Lab dependency pin without the inherited FPGS/P25 validation gates.

## What a further 2× actually requires

The original 16K Kuka physics graph takes approximately 17.877 ms on RTX and
17.723 ms on GB300. Selected constraint kernels account for interval unions of
9.393 and 9.533 ms respectively, with overlap against unchanged work. Even an
optimistic subtraction leaves only 56.82/83.95 microseconds per solver call for
the **entire replacement constraint path** if everything else stays unchanged.
There are eight solver calls per environment step.

Pure canonical allocation/finalization alone costs approximately 52.23/45.92
microseconds per call. Keeping allocation, metadata, Jacobian maintenance and
difficult-world solves therefore cannot be called a standalone 2× architecture.
The next successful design must remove producers as well as consumers, or
combine a large constraint gain with a separate broad dynamics/collision gain.

A 4× gain is impossible by changing only this constraint-stage selection:
the optimistic unchanged remainder is already about 8.48/8.19 ms, well above
the 4.47/4.43 ms whole-step target. This is a scope bound, not a claim that the
remaining work is intrinsically irreducible. A proposed 2–4× architecture must
budget its dynamics, collision, allocation, fallback and public outputs together.

The implementation north star remains compact, topology-specialized world work:
shared factor coordinates, no unnecessary dense response materialization,
state kept near its consumer, and work proportional to active constraints.
The experiments below show that simply putting everything in one kernel does
not achieve that goal: serialized work, large per-thread state and duplicated
producer work can make a fused implementation substantially slower.

## Structural routes investigated and closed

| Route | Evidence | Decision |
| --- | --- | --- |
| Tree-order factor/whitening and resident sweeps | Sparse algebra is correct; modest isolated packet gains did not establish the required whole-scope improvement | Close mappings; do not tune their launch parameters |
| Nonlinear active-set/contact-block solver replacements | Some typical residuals improve; hard physical tails or complete GPU cost fail | No recipe promotion or hidden extra iteration allowance |
| Cooperative Kuka dynamics + rows + solve + publication | Complete component cost averages 6.486 ms RTX / 4.759 ms GB per solver call | Far beyond whole-step budget; close mapping |
| Raw-row exact-zero classifier | Classifier alone costs roughly 336–338 µs RTX / 462 µs GB per call, before omitted runtime work | Close mapping |
| FP64 body-twist certificate | Correctness passes, but four-kernel screen costs 2138/2160 µs RTX and 2786/2815 µs GB for refresh/reuse | More expensive than the original entire constraint stage; close mapping |
| Strict outward-FP32 certificate | Complete screen costs 1094/1113 µs RTX and 1534/1560 µs GB for refresh/reuse | Still consumes nearly all or more than the original constraint stage; close mapping |
| Multi-position-limit exact projection extension | At most 620 of 65,536 worlds, about 3.1–3.5% of the total same-eight row-visit proxy | Too little whole-step scope; close before another native solver |
| Certified collision rejection | Promising isolated collision work; full-size private quality oracle remains unresolved | Preserve failure and do not promote component results |

These are failures of particular mappings or quality gates, not proofs that
tree algorithms, active sets or collision coherence cannot provide large gains.
Every quoted component excludes no work silently; excluded scope is recorded
in its detailed local artifact. No table entry is an accepted whole-step win.

## Closed interval-screen predecessor

For a cold-start world, first propose either unchanged predictor velocity or
the exact physical projection of one violated position limit using the
**original held** `L23 L23ᵀ`. The GS denominator stabilizer CFM is not added to
the physical projection operator. This closed proposal proves unique FP32 rounding
cells for published velocity and impulse and bounds their momentum error.

Then prove all original active position-limit rows, all raw responsive contacts and all
six free-body speed limits. Contact proof uses body twists rather than building
every generalized contact Jacobian. Any unsupported input, uncertain bound,
nonfinite arithmetic or violated physical row sends the **whole world** through
the unchanged original cold eight-sweep solve. There is no reused certificate
and no additional iterative solve hidden in the proposal.

The CPU outward-FP32 equation oracle certifies 60,695 of 65,536 sampled Kuka
worlds, with no source-row false positives across 700,234 contact normals.
That is a coverage result, not a speedup or a broad-task claim. A separate
AnyMal/Allegro census found negligible zero/one applicability, so this shortcut
must not be advertised as helping those environments.

The experimental Newton controller currently preserves canonical allocation,
row metadata, force identities and full public outputs. Only private compute
counts/flags select omitted work. The actual original masked-J writer passed
paired 512-world eager/graph routing tests: fallback writes remained exact and
only certified contact-J cells remained unmaterialized. Complete integrated
physics and whole-step performance still require their own gates.

### Compiler boundary is part of numerical correctness

The FP32 carrier retains outward rounding and the original source-operation
roundoff/FTZ envelopes; it does not widen a physical acceptance tolerance.
The intended directed FMA behavior is documented in
[CUDA 12.9.1](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html).

Actual fast-math testing exposed a failed assumption: source-level integer
bit manipulation was optimized back to `neg.ftz.f32`, turning a negative
subnormal endpoint into negative zero. The failed source and tests are retained.
A strict sibling uses three opaque integer PTX primitives for bit extraction,
negation and magnitude; geometry and physical/source envelopes are unchanged.
Both GPUs pass eleven exact edge cases in separately compiled fast-math-off
and fast-math-on modules, including eager and two graph replays per mode.
See NVIDIA's [inline PTX compiler contract](https://docs.nvidia.com/cuda/archive/12.9.1/inline-ptx-assembly/index.html#incorrect-optimization).

Strict native 512-world checks also pass on both cards, with 477/472 certified
worlds on RTX and 469/466 on GB for the adjacent refresh/reuse captures. These
checks include complete raw output ownership, original source-row feasibility,
independent physical equations, graph replay, overflow and zero-normal fallback.
The full-size strict gate also passes its numerical and ownership checks, but
fails the architecture cost budget: 1094.016/1113.184 µs RTX and
1534.464/1560.064 µs GB for refresh/reuse. These timings include all four
proposal/body/raw-contact/finalize kernels. The disabled arm is only boundary
overhead, not a physics baseline. The final actual timed outputs are saved and
rechecked; an eager replay is not substituted for timed output validation.
There is no live whole-physics test or launch-parameter tuning of this closed
mapping.

Source-ordered Nsight attribution explains why a contact-only rewrite is also
insufficient. In two cost-phase eager controls, proposal plus body preparation
alone takes 788–836 µs RTX and 933–969 µs GB. Raw contacts add approximately
304–310/544–550 µs. These traces have aggregate graph timings, **not** forty
per-kernel graph samples; the stage figures are explicitly eager diagnostics.
The profiler also warns about the newer driver, and eight early GB proof-graph
activity records are missing, although their launch APIs and all 204 cost-graph
activities are present. The unprofiled complete-cost results remain authoritative.

A body-affine representation can replace hundreds of directed operations per
contact by ordinary point evaluation plus rigorous body-owned error bounds.
However, it must replace the current proposal/body producers as well. The first
CPU world-max-radius census established coverage headroom only: its bounds
were derived from the expensive oracle. The later constructive study below
removes that dependency, but still needs a new native body/contact owner.

### New direction: direct FP32 candidate, independent residual proof

The final structural experiment separates candidate generation from acceptance.
It replaces the FP64 interval triangular solve and unique-rounding-cell demand
with an ordinary FP32 direct solve, then two parallel directed-FP32 triangular
matrix-vector products that enclose the **actual published** momentum residual
`L * (L.T * (velocity - predictor)) - signed_impulse`. The selected coordinate
is pinned to its exact represented limit equality. This is neither an extra
GS sweep nor iterative refinement, and it does not introduce a relaxed physical
tolerance. There is deliberately no acceptance mask yet.

The full CPU census accounts for 65,536 worlds and 8,302 one-limit candidates.
Of those, 8,132 satisfy every original unilateral row; 170 fail at least one.
Five GB candidates also fail another position limit; all failures remain
recorded. The preserved CPU report has a stale scope label saying “Four
actual512”; its four fixture paths and `worlds=16384` records identify the
actual full-size census. The largest measured
velocity difference from the FP64 ideal projection is about 1.18e-7. A small
momentum residual alone is not a forward-error guarantee: conditioning and
publication rounding must be included. The distinction follows standard
[triangular-solver error analysis](https://eprints.maths.manchester.ac.uk/338/1/0916025.pdf);
our contact-specific acceptance still needs its own proof.

A separate seed-one CPU check on four retained 512-world captures adds 237
one-limit candidates: all satisfy the original position limits, and 233
satisfy all original unilateral rows. The four contact failures stay rejected.
Maximum observed velocity error is 9.414e-8 and momentum defect 1.067e-7.
This is additional saved-state validation, not a sustained rollout or a
native/full-pipeline quality acceptance.

A constructive, directed-FP32 comparison-factor bound supplies an upper bound
on `norm(inv(H), 2)`. Combined with the residual enclosure, it bounds velocity,
impulse and energy error without choosing one identical rounded solution.
This factor-epoch bound has a real computation/ownership cost; it is not
silently assumed to exist for free. An independent CPU audit consumes the saved
native residual intervals verbatim and rechecks all 6,095 selected momentum
components with exact rational arithmetic. The resulting hull preserves all
262 previously feasible cases across the four 512-world captures; all 265
exact-projection impulses are certified positive. The three point-infeasible
candidates stay rejected. The maximum certified velocity-error bound is about
3.02e-5, not the much smaller observed error substituted as a proof radius.

The new native candidate/residual primitive passes paired RTX/GB 512-world
checks, including a negative-subnormal factor coefficient, separately compiled
fast-math-off/on modules, eager execution and two graph replays. Every selected
momentum component is checked with exact rational arithmetic against the actual
published outputs. Native source SHA256:
`750c0c2862d3198b2204013a403b73bf4f54a22ca864a9d47ddb50396a4882b9`.
This is a proof primitive, not an accepted complete solve. The inverse bound,
raw-contact/body certificate, canonical allocation, fallback and full runtime
integration remain outside that native kernel.

Its final paired 16K component gate passes: **34.048/34.352 µs RTX and
35.168/35.312 µs GB** for refresh/reuse. Both complete 16K rational/graph checks
precede event creation; fifty balanced enabled/disabled rounds discard ten
and retain forty samples per arm. Actual last timed outputs match the proved
buffers without an eager repair, with exact same held-factor allocation on
reuse. The disabled arm is boundary overhead, not a solver baseline. These
are timings of candidate generation plus its directed momentum residual and
all native output stores—not complete acceptance, proof or whole physics.
The old proposal attribution was obtained from eager profiler controls, so
do not turn those differently scoped measurements into a controlled speedup.

The first paired cost attempt completed its numerical/timing phases but failed
to write the final reports: original metadata contains an allowed positive
infinity, rejected by strict outer JSON serialization. Its logs/raw outputs
remain preserved and its console medians are not accepted evidence. A new
report-only sibling retains exact original metadata JSON as a string plus
SHA256; its numerical/event functions are unchanged. The second run has full
samples, output hashes, source guards, child reaping and final idle checks.

### Constructive affine contact screen: coverage, not native performance

A new CPU producer builds body affine point-velocity maps and positive
roundoff bounds from current motion subspaces, body transforms and the
candidate velocity hull. Contact evaluation uses an ordinary FP32 center plus
those body/world bounds. No observed oracle maximum or fitted tolerance is
used to construct them. Original source-rounding error, world-coordinate
translation, physical geometry, margin arithmetic and both bias laws remain
charged. This study uses the CPU candidate/residual envelope, not a complete
native implementation of the pipeline above.

The same frozen constructive arithmetic passes all four 16K captures:
58,051 of 65,536 worlds (88.58%), comprising 50,536 zero-limit and 7,515
one-limit worlds. All 700,234 responsive contacts are accounted for, with zero
source/physical enclosure failures or false positives in the admitted domain.
The domain is explicit and conservative: bounded contact points, margins and
normal magnitude, the original fixed timestep, stationary prescribed table,
no shared-anchor mode, and finite positive applicable MF depenetration caps.
Any domain failure rejects the complete world. Native compiler qualification,
full runtime integration and trajectory quality remain untested.

An independent census attributes only 74.66% of the same-eight row-visit
proxy to these worlds. All 1,152 mixed-MF snapshot worlds remain fallback;
their source free bodies have positive-infinite depenetration caps, which this
prototype deliberately does not admit. A future proof of the inactive-cap
identity is useful only if it enables a materially cheaper MF dispatch—not
as a percent-level coverage exercise. If the original MF exclusive work stays
unchanged while every other selected constraint node were free, the optimistic
whole-step ceiling is only 1.682× RTX / 1.652× GB. Visit counts are not timings,
and this node-subtraction ceiling is not a dependency-derived guarantee.

The next architectural gate must therefore include the new candidate,
factor-error bound, body maps, raw contacts, allocation/publication and complete
fallback together. The simple body-map layout alone publishes 21 floats per
body (44.0 MB over 16K worlds), before other outputs. Neither that work nor
the difficult-world tail can be treated as free.

### Decision: remove runtime work, and separate proof from numerical validation

The rigorous comparison bound need not be built for every world. A stateless
implementation can compute it only for current one-limit candidates from the
same held factor used by their proposal; zero-correction worlds do not need
it. This avoids introducing a factor-generation cache. A later cache would
need invalidation on actual device factor writes, not a host step counter;
272 RTX and 297 GB worlds first become one-limit candidates on the reuse
capture, so “mass reused” cannot mean “bound already present.”

A concrete next owner keeps the original fast dynamics/factor stages and
combines the candidate, optional bound, tree-level body twists, affine maps
and complete raw-contact scan within one world CTA. A logical shared layout
is roughly 9 KiB and avoids the separate 44 MB map publication. This is a
layout estimate, not an occupancy or timing prediction. Complete raw-contact
bins must be produced after collision completion, with exact original IDs,
generation/storage checks and whole-world overflow fallback. Public contact
geometry, force semantics and state outputs must remain correct, but the late
consumer audit below removes the earlier assumption that every internal
canonical row must be materialized. Allocation and difficult-world work must
be redesigned or explicitly charged. The slow all-dynamics monolithic mapping
is not the implementation to revive.

The user's requirement is correct numerical convergence—not a per-operation
formal certificate. Outward source arithmetic, an on-device inverse bound
and proof of a nearby exact solution are stronger optional guarantees. They
must not become an accidental requirement that blocks ordinary floating-point
solver architectures. A new approximate FP32 runtime may instead use complete,
dimensionally scaled feasibility/complementarity/momentum screening, with the
rigorous machinery retained as an offline reference. It must declare that
different numerical contract honestly, preserve full fallback/ownership and
the fixed iteration allowance, and pass held-out conditioning/contact tails,
reset paths and sustained trajectory tests. Thresholds cannot be widened after
seeing failures, and no present experiment establishes those gates.

The next effort should target this complete producer/allocation/fallback
boundary or an independent broad dynamics/collision gain. Stop it on failed
whole-scope cost or held-out physical quality; do not spend another cycle
tuning the closed screen or promoting its component ratios.

### Late scope correction: row allocation is an implementation choice

The original FPGS `update_contacts` converts internal row impulses into
`Contacts.rigid_contact_force` and optional spatial `Contacts.force`.
Newton's `SensorContact` and Isaac Lab's contact sensor consume those public
forces and raw collision geometry—not FPGS row arrays. Final integration
consumes solved velocity, not canonical row metadata. Therefore a zero-contact-
impulse world can publish zero forces directly without constructing all its
dense/MF rows. A selected joint-limit impulse changes solved velocity; it must
not be added to external `body_f` or `Control.joint_f` inputs.

This is a source-backed permission to redesign the representation, not a
measured deletion of 52 µs. The zero-force branch must run before reading stale
row slots and actively overwrite every active force record. Collision count,
geometry, original contact IDs and sensor behavior remain unchanged. A contact
generation can span several solver substeps, so force validity must also refer
to the last solve and its timestep. Fallback must rebuild fresh rows after a
bypass. Internal maintenance, warm-start modes and honest row-watermark/debug
reporting require explicit handling; the current cold Kuka benchmark does not
enable row watermarks by default.

The separate MF audit also prevents a false shortcut: dense-only worlds already
return before expensive MF staging/sweeps. The active mixed-world kernel solves
all dense and MF rows together, with long serial dependencies. A compact queue
can reuse an existing world-ID interface, but current traces do not show how
much time compaction would save. A larger mixed-world owner must preserve held
dense factors versus current MF free-body inertia, complete physical laws and
the same iteration allowance. Moving a grid boundary alone is not the proposed
2× algorithm.

### Runtime lifecycle

Original Lab startup/reset notifications occur after initial solver capture.
The Newton-only controller revokes its original device admission allocation
before a notification and re-admits only after unchanged static topology,
scalar recipe, known backing-buffer sets and registered graph bindings are
verified. Numeric material/mass/armature updates remain fresh proof inputs.
Structural or storage changes stay disabled; original callback failures do not
restore admission. Both eager and newly captured proof launches bind the same
revocable allocation, not a replaceable Python attribute. No Lab scheduler
override is used.

This lifecycle has focused CPU tests but has not passed an integrated live
controller test. Exact re-admission currently entails up to thirty synchronous
host reads, approximately 46.5 MiB of topology data at 16K worlds per relevant
notification. That is a source-derived volume, not a measured latency. The
`MODEL_PROPERTIES` contract covers more than gravity, so silently replacing
these checks with pointer equality is not justified. Reset-inclusive runtime
cost is a separate unresolved architecture gate.

The later original-API audit narrows this obligation for a future implementation:
ordinary FPGS builds an immutable constructor topology plan and does not
authenticate arbitrary topology writes on every property notification. A new
fast mode can match that supported mutation/buffer-lifetime contract instead
of inheriting the old controller's stronger static-byte checks. Structural or
incompatible storage changes still require reconstruction/recapture; supported
drive/limit/kinematic updates still require fresh recipe checks or fallback.
This is not unconditional re-admission, and the old controller's stronger
claims/tests have not been retroactively weakened.

The earlier routing reports retain hashes of historical controller/solver
versions whose exact source bytes were not separately preserved. Their actual
native routing sources and output artifacts are retained, but those historical
controller snapshots cannot be independently reconstructed. Do not present
the boundary test as complete current-controller integration evidence.

### Cache provenance correction

The source harness resets once before warmup, not again before capture. The
16K fixtures follow 200 random-action environment steps; only 3 RTX / 7 GB
worlds have all hand velocities zero. The two initial cache-validity arrays
are zero then one for the seed-zero 16K captures, but seed-one captures start
fully valid. A small reset subset can invalidate the global FK cache through
the gravity notification; the exact preceding notification IDs were not captured.
Deleting even all measured prefix FK work would give only an optimistic
1.031× RTX / 1.021× GB whole-physics improvement. That route is closed as too
small for this mission. Final FK and public body outputs cannot simply be omitted.

## Broader environments and MJWarp

This window establishes no new accepted cross-environment or MJWarp speedup.
The [original RTX 5090 handoff reproduction](README.md#fresh-measurements)
corrected Franka to approximately 3.7–3.9×, not 5.1×. Those historical numbers
are not measurements on the current pair of cards. The subsequent RTX/GB
Franka ratios of 8.76×/2.23× in the
[earlier architecture report](ARCHITECTURE_UPDATE_20260911.md#fresh-mjwarp-comparison)
include warning-heavy MJWarp runs and do not demonstrate an accuracy-matched
near-10× solver advantage. SO101 keyboard timing also remains separated from
accuracy acceptance: its row-capacity drops and MJWarp warnings are unresolved.

## Reproduction and handoff status

The closed core is retained locally at
`/home/octi/Projects/fpgs-large-gain-evidence-20260911-ri5D08/closed-core`:
5,202 files, 11,193,762,719 bytes, individually hashed before and after copying.
Its `MANIFEST.json` SHA256 is
`91b244a974bf7803b9f9c83847ba53ae9601587d73cfdcaaa837c183f5e0bfa5`.
It includes the actual native sources, complete strict/DP proof and cost
outputs, failed compiler cases, original captures, and default-off experimental
Newton trees. `.git`, `.venv` and Python caches are excluded; omitted symlinks
are listed. Repository HEADs, dirty status and tracked diffs are recorded.
This archive does not claim to recover missing historical source bytes.

Additional frozen sections under the same archive root are:

| Section | Files | Bytes | Manifest SHA256 |
| --- | ---: | ---: | --- |
| `closed-attribution` | 78 | 773,468,026 | `cc6934e9b5696c1ecf1948cf8d518237afb1a5a1a7fad660f3d98e9e1b474346` |
| `closed-dependencies` | 11,217 | 25,529,476,935 | `b52a1f995a886e889de6f597376566829d431761a05ea3d4f75ca3871e7dff66` |
| `new-proposal` | 310 | 89,534,857 | `5284f3f39ace35bde470beede91732ffba327d34e3a01ea6d31693c649b852fd` |
| `final-scope` | 9 | 313,831 | `c1b73646ef7e5766bc9b5fbee4e66cc1da08e6e222c3cf9add5b6e3d1c4d7236` |

`new-proposal` includes the direct candidate source, numerical contracts,
constructive affine studies, independent audits, complete paired 512 and 16K
outputs, the failed cost-report attempt and its separate reporting-only repair.
All five sections pass a full `sha256sum -c SHA256SUMS` recheck after copying.
`final-scope` contains the public-force/row-representation and MF-dispatch
audits plus the separate-seed CPU check. It corrects stronger assumptions in
earlier design notes without rewriting their historical source or results.

The [paired evidence launcher](reproduce_large_gain_20260911.py) verifies retained
sources by default and can rerun strict compiler edges, native 512-world checks
or complete 16K screen cost. It requires the recorded local paths/environment
and GPU UUIDs; it is not a portable fresh-install reproduction. The original
paired Lab harness remains unchanged. No command promotes the rejected mode.

```bash
uv run --no-project --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python \
  python reports/fpgs/reproduce_large_gain_20260911.py
uv run --no-project --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python \
  python reports/fpgs/reproduce_large_gain_20260911.py --run edges \
  --output /tmp/fpgs-strict-edge-reproduction-new
```

Use `--run native512` or `--run cost16k` with a different new output directory
for the other gates. Cost16K requires both preserved native512 predecessor
reports and reruns full-size correctness before timing. Each GPU is an
independent process; outputs, errors and source identities stay separate.
Verify all archived bytes with `sha256sum -c SHA256SUMS` inside each archive
section. Large reference captures stay local; no capture is relabeled as
steady-state whole-physics timing.

The new direct primitive uses private adapters around the same paired process
owner; it is not one of the older strict-screen launcher's built-in modes:

```bash
uv run --no-project --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python \
  python /tmp/fpgs-posteriori-paired-iivr8u/run_pair.py --run proposal512 \
  --output /tmp/fpgs-direct-proposal512-reproduction-new
uv run --no-project --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python \
  python /tmp/fpgs-posteriori-cost-paired-314WPA/run_pair_reportfix.py --run proposal16k \
  --output /tmp/fpgs-direct-proposal16k-reproduction-new
```

The cost adapter requires the exact retained native512 predecessor reports
and rechecks both full 16K fixtures before timing. The accepted component
record is `cost16k_02/paired.json`, SHA256
`a131443eba67dad25eb78b44df7097c14cdd000dde4d33d883ccc2c7f9a1630b`.
Original absolute paths, environment and GPU UUIDs are intentional prerequisites;
the archive is not a portable installed environment. Formatting installed
development-tool dependencies in the existing virtual environment, but its
Python 3.12.14, Warp 1.17.0, Torch 2.11.0+cu130 and Newton runtime pins did not
change. No virtual-environment byte-identity claim is made.

## Handoff branch and validation

Both forks use branch `ooctipus/fpgs-large-gain-20260911`. Newton's exact
documentation checkpoint is
[`fe484a0a58293925f889cb8c641e781569c9ae70`](https://github.com/ooctipus/newton/commit/fe484a0a58293925f889cb8c641e781569c9ae70),
derived from `a7eb7d15589a3c6032288a488463eb8e028d88e6` with original handoff
`31cf87f4694f873a027e41e2ca5e9ad441234456` preserved in its ancestry. This Lab
branch retains its handoff/optimization reports and `2129e5628` ancestry.
Neither branch enables the new research kernels or advances the blocked
Newton dependency in `pyproject.toml`/`uv.lock`.

Required Newton `uvx pre-commit run -a` and Lab `uv run isaaclab -f` checks
pass. Focused CPU numerical, ownership, report-failure and reproduction tests
pass, along with the paired native/cost gates described above. These checks
do not replace the unrun integrated row-free solver, sustained physical-quality
tests, or inherited FPGS/P25 dependency gates. Original runtime worktrees remain
clean at their exact pins. Large local evidence includes failures, not just
successful runs; the final local handoff index records the published commits,
archive manifests and closure status.
