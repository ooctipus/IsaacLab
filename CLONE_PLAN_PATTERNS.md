# Canonical clone-plan resolution patterns

Reference for backend authors writing or reviewing asset / sensor
`_initialize_impl` methods.

The active [`ClonePlan`](source/isaaclab/isaaclab/cloner/clone_plan.py) is the
single source of truth for which envs exist, which source prims back them, and
where each source's destination prims live. Backend code should resolve
topology **source-side** through the helpers in
[`isaaclab.cloner.cloner_utils`](source/isaaclab/isaaclab/cloner/cloner_utils.py),
not by globbing destination prims on the live USD stage.

Two reasons this matters:

1. **Newton-replicated stages don't author destination prims.** Code that
   reaches for `find_first_matching_prim("/World/envs/env_.*/...")` to learn
   topology fails or silently zeros out at init time.
2. **Heterogeneous variant clones are first-class** in the plan. Stripping
   `env_0` out of paths and re-globbing produces wrong answers when different
   envs receive different prototypes.

## The three helpers

| Helper                                            | Walks                       | Returns                                          |
| ------------------------------------------------- | --------------------------- | ------------------------------------------------ |
| `source_prim(plan, expr)`                         | none — pure lookup          | first `(prim, source_root, destination, env_ids)` or 4-tuple of `None` |
| `descend_source_prims(plan, expr, predicate)`     | self + descendants          | `list[(prim, source_root, destination, env_ids)]` |
| `ascend_source_prims(plan, expr, predicate)`      | self + ancestors up to `source_root` | `list[(prim, source_root, destination, env_ids)]` |

`predicate` is **required** on the walking pair — without a filter you have
no walk, just a lookup, and that's what `source_prim` is for.

## Where the validation gate already lives

The base classes `AssetBase` and `SensorBase` already gate initialization on
plan coverage, e.g.:

```python
# AssetBase.__init__
if source_prim(plan, self.cfg.prim_path)[0] is None:
    raise RuntimeError(f"Asset '{self.cfg.prim_path}' is not covered by the active ClonePlan.")
```

Subclasses **must not** re-emit a "is this path valid" check. They only
validate predicate-specific conditions (e.g. "exactly one
`ArticulationRootAPI` under this path").

Sensors receive the active plan as `self._clone_plan` after
`super()._initialize_impl()` runs. Assets fetch it via
`SimulationContext.instance().get_clone_plan()`.

---

## Pattern A — single source prim → per-env glob

**When to use it.** You need a wildcard expression that matches one prim per
environment, typically to feed a `physics_sim_view` view constructor. This is
the dominant pattern across PhysX/OvPhysx assets and sensors.

```python
plan = SimulationContext.instance().get_clone_plan()
roots = descend_source_prims(plan, self.cfg.prim_path, lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI))
if len(roots) != 1:
    raise RuntimeError(f"Expected one RigidBodyAPI prim under '{self.cfg.prim_path}'; found {len(roots)}.")
root_prim, source_root, destination, _ = roots[0]
view_glob = destination.format("*") + root_prim.GetPath().pathString[len(source_root):]
self._root_view = self._physics_sim_view.create_rigid_body_view(view_glob)
```

**Notes.**

- Build the glob from `destination.format("*")` directly — do not append
  `.replace(".*", "*")` afterwards. Plan destinations carry `"{}"`, not `".*"`.
- The relative suffix is computed once with
  `root_prim.GetPath().pathString[len(source_root):]`. This works for both
  "root_prim is at source_root" (suffix = `""`) and deeper hits.
- Predicates compose: pass `lambda p: p.HasAPI(X) and not p.HasAPI(Y)`
  rather than chaining helpers.

**Used in.**

- `isaaclab_physx/.../articulation/articulation.py`
- `isaaclab_physx/.../rigid_object/rigid_object.py`
- `isaaclab_physx/.../rigid_object_collection/rigid_object_collection.py`
- `isaaclab_physx/.../deformable_object/deformable_object.py`
- `isaaclab_physx/.../surface_gripper/surface_gripper.py`
- `isaaclab_physx/.../sensors/joint_wrench/joint_wrench_sensor.py`
- `isaaclab_ovphysx/.../articulation/articulation.py`
- `isaaclab_ovphysx/.../rigid_object/rigid_object.py`
- `isaaclab_ovphysx/.../rigid_object_collection/rigid_object_collection.py`

---

## Pattern B — ancestor walk for a surrounding physics body

**When to use it.** The configured `cfg.prim_path` may live inside a physics
body and the sensor needs both the surrounding rigid body's view *and* the
fixed sensor-to-body offset (IMU, PVA).

```python
imu_prim, *_ = source_prim(self._clone_plan, self.cfg.prim_path)
bodies = ascend_source_prims(self._clone_plan, self.cfg.prim_path, lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI))
if not bodies:
    raise RuntimeError(f"No rigid body ancestor under: {self.cfg.prim_path}")
body, source_root, destination, _ = bodies[0]
self._rigid_parent_expr = (destination + body.GetPath().pathString[len(source_root):]).replace("{}", "*")
self._view = self._physics_sim_view.create_rigid_body_view(self._rigid_parent_expr)
fixed_pos_b, fixed_quat_b = (
    (None, None) if body == imu_prim else sim_utils.resolve_prim_pose(imu_prim, body)
)
```

**Notes.**

- `ascend_source_prims` returns matches closest-first and stops at the plan
  entry's `source_root`. Going above `source_root` would yield prims outside
  the per-asset frame and break the view-glob construction.
- The `body == imu_prim` check is the "sensor is already mounted at the rigid
  body" short-circuit; it skips the offset computation and yields a trivial
  identity transform.
- Two helper calls (one `source_prim`, one `ascend_source_prims`) is
  intentional — the lookup gives you the sensor frame, the ascend gives you
  the body frame and the destination for view-glob construction.

**Used in.**

- `isaaclab_physx/.../sensors/imu/imu.py`
- `isaaclab_physx/.../sensors/pva/pva.py`
- `isaaclab_physx/.../sensors/ray_caster/ray_caster.py` (tracked branch — falls back to Pattern D when `ascend_source_prims` returns `[]`)

---

## Pattern C — boundary adapter for non-plan-aware APIs

**Default inside an asset/sensor: don't enumerate.** The plan-native shape is
`(destination, env_ids_from_mask)` — a destination string with a `"{}"` slot
plus a small set of int env ids. If your code is iterating per env, build the
path in the loop:

```python
plan = SimulationContext.instance().get_clone_plan()
src_idx = plan.sources.index(matching_source_root)
covered_env_ids = plan.clone_mask[src_idx].nonzero(as_tuple=False).flatten().tolist()
for env_id in covered_env_ids:
    path = destination.format(env_id)        # one path at a time, no allocation
    ...
```

**Use `expand_clone_plan_paths` only when a downstream API insists on a flat
`list[str]`.** This is a boundary adapter, not a canonical inside-the-asset
shape. Legitimate cases:

- PhysX returns concrete prim paths from `create_rigid_body_view([globs])` and
  you need to invert that to "which `(env, body)` is this?" — see
  `frame_transformer.py`.
- A renderer / visualization API takes `list[str]` of camera paths — see
  `camera.py` and `ovrtx_renderer.py`.

```python
paths = expand_clone_plan_paths(plan, prim_expr)  # list[str | None] of length num_envs
cam_paths = tuple(p for p in paths if p is not None)
```

**Notes.**

- Length is always `plan.clone_mask.shape[1]`. Uncovered envs get `None`;
  callers either skip or raise.
- `expand_clone_plan_paths` raises if two plan entries claim the same env
  (ambiguous destination). That's an integrity check, not an expected case.

### When NOT to use it (anti-pattern)

If your code looks like this, you don't need Pattern C — you need the destination:

```python
# Anti-pattern: pre-allocates list[str | None] then immediately re-indexes it.
expanded = expand_clone_plan_paths(plan, entry.vis_mesh_prim_path)
for inst_idx, offset in enumerate(entry.particle_offsets):
    vis_prim = stage.GetPrimAtPath(expanded[inst_idx])
    ...

# Plan-native: same end state, no intermediate list, scales to large num_envs.
for inst_idx, offset in enumerate(entry.particle_offsets):
    vis_prim = stage.GetPrimAtPath(destination.format(inst_idx))
    ...
```

The anti-pattern currently lives in
`isaaclab_contrib/.../deformable/{vbd_manager, coupled_*_vbd_manager}.py` —
flagged for follow-up cleanup, not in scope of the clone-plan migration PR.

### Worked examples

The three test fixtures in
[`test_cloner.py`](source/isaaclab/test/sim/test_cloner.py) double as the
canonical input/output spec.

**Example 1 — homogeneous plan, full coverage.**

```python
# Inputs
plan = ClonePlan(
    sources=("/World/source/Robot",),
    destinations=("/World/envs/env_{}/Robot",),
    clone_mask=torch.ones((1, 3), dtype=torch.bool),     # 3 envs, all covered
)
prim_expr = "/World/envs/env_.*/Robot/base"

# Output
expand_clone_plan_paths(plan, prim_expr) == [
    "/World/envs/env_0/Robot/base",
    "/World/envs/env_1/Robot/base",
    "/World/envs/env_2/Robot/base",
]
```

The output is a length-`num_envs` list (3 here). The query's `env_.*` slot is
substituted with each env id, and the `/Robot/base` suffix carries through
because the plan covers `/Robot` and the suffix is an unambiguous descendant.

**Example 2 — heterogeneous variant clones, partial coverage.**

```python
# Inputs
mask = torch.zeros((2, 4), dtype=torch.bool)
mask[0, [0, 1]] = True
mask[1, [2]]    = True                                    # env 3 deliberately uncovered
plan = ClonePlan(
    sources=("/World/source/RobotA", "/World/source/RobotB"),
    destinations=("/World/envs/env_{}/RobotA", "/World/envs/env_{}/RobotB"),
    clone_mask=mask,
)

# Outputs (one query per variant, against the same plan)
expand_clone_plan_paths(plan, "/World/envs/env_.*/RobotA/base") == [
    "/World/envs/env_0/RobotA/base",
    "/World/envs/env_1/RobotA/base",
    None,                                                 # env 2 has RobotB, not RobotA
    None,                                                 # env 3 is uncovered entirely
]

expand_clone_plan_paths(plan, "/World/envs/env_.*/RobotB/base") == [
    None,
    None,
    "/World/envs/env_2/RobotB/base",
    None,
]
```

This is the case where the old `path.replace("env_0", "env_*")` idiom silently
returned wrong answers: it would have produced `env_*/RobotA/base` for every
env regardless of which variant actually occupies that env. Pattern C surfaces
the truth (`None` for envs that don't have this variant) so the caller can
either skip them or raise.

**Example 3 — three-way variant split.**

```python
# Inputs
mask = torch.zeros((3, 4), dtype=torch.bool)
mask[0, [0, 1]] = True
mask[1, [2]]    = True
mask[2, [3]]    = True
plan = ClonePlan(
    sources=("/World/source/Robot",) * 3,                 # same prototype, different dest groups
    destinations=(
        "/World/envs/env_{}/RobotA",
        "/World/envs/env_{}/RobotB",
        "/World/envs/env_{}/RobotC",
    ),
    clone_mask=mask,
)

# Output for the third variant
expand_clone_plan_paths(plan, "/World/envs/env_.*/RobotC/base") == [
    None, None, None, "/World/envs/env_3/RobotC/base",
]
```

Each variant exposes a different destination; the query selects which
one, and the mask determines which env (if any) actually has it.

**Edge case — double coverage raises.**

```python
plan = ClonePlan(
    sources=("/World/envs/env_0", "/World/envs/env_0/Object"),
    destinations=("/World/envs/env_{}", "/World/envs/env_{}/Object"),
    clone_mask=torch.ones((2, 2), dtype=torch.bool),
)
expand_clone_plan_paths(plan, "/World/envs/env_.*/Object/Body/Camera")
# RuntimeError: env 0 matched twice for '...' (existing '...', new '...').
```

This catches plan-construction bugs where two entries (e.g., a parent env
destination and a child object destination) both claim ownership of the same
path expression. It's an integrity check, not a normal-flow code path.

**Used in — genuine boundary cases.**

- `isaaclab_physx/.../sensors/frame_transformer/frame_transformer.py` —
  inverts PhysX `view.prim_paths` back to per-env body indices.
- `isaaclab/isaaclab/sensors/camera/camera.py` — feeds the renderer-side
  `CameraRenderSpec`, which takes `list[str]`.
- `isaaclab_ov/.../renderers/ovrtx_renderer.py` — OmniRTX scene-partition
  tokens consume concrete paths.

**Used in — should be ported (lazy enumeration).**

- `isaaclab/isaaclab/assets/asset_base.py::set_visibility` — caches
  `[stage.GetPrimAtPath(p) for p in expand_clone_plan_paths(...)]`. Equivalent
  to `[stage.GetPrimAtPath(destination.format(i)) for i in covered_env_ids]`.
- `isaaclab/isaaclab/envs/utils/camera_view.py` — currently compares against
  expanded paths; could match by `(destination, env_id)` if the camera registry
  switched its key.
- `isaaclab_contrib/.../deformable/{vbd_manager, coupled_*_vbd_manager}.py` —
  the anti-pattern shown above.

---

## Pattern D — pose composition for non-physics frames (Win A)

**When to use it.** A sensor's frame is not driven by the physics engine and
there is no destination prim to read a transform from. Compose the per-env
world pose from the plan's `env_pose` and the source-relative pose:

```python
prim, source_root, _, env_ids = source_prim(plan, self.cfg.prim_path)
rel_pos, rel_quat = sim_utils.resolve_prim_pose(
    prim, sim_utils.get_current_stage().GetPrimAtPath(source_root)
)
env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=plan.env_pose.device)
env_pose = plan.env_pose[env_ids_t]
world_pos, world_quat = combine_frame_transforms(
    env_pose[:, :3], env_pose[:, 3:],
    torch.as_tensor(rel_pos, dtype=env_pose.dtype, device=env_pose.device).expand(len(env_ids), 3),
    torch.as_tensor(rel_quat, dtype=env_pose.dtype, device=env_pose.device).expand(len(env_ids), 4),
)
```

**Notes.**

- `plan.env_pose` is `[num_envs, 7]` with `xyz` position followed by `xyzw`
  quaternion.
- `resolve_prim_pose(child, parent)` returns the pose of `child` expressed in
  the frame of `parent`.
- Index `env_pose` with the matched `env_ids` rather than assuming all envs
  are covered.

**Used in.**

- `isaaclab_physx/.../sensors/ray_caster/ray_caster.py::_initialize_static_pose_tracking`
- `isaaclab/isaaclab/envs/utils/camera_view.py::prim_world_positions`

---

## Pattern E (escape hatch) — raw plan iteration

**When to use it.** You need to enumerate plan entries directly without doing
source-prim resolution — for instance, registering Newton sites where the
"prim" is a label, not a USD prim.

```python
for source_root, destination, source_path, env_ids in iter_clone_plan_matches(plan, prim_expr):
    ...
```

`iter_clone_plan_matches` is the building block the other helpers are written
on top of. Reach for the higher-level helpers first; only drop down to this
when you have a reason to.

**Used in.**

- `isaaclab_newton/.../sensors/ray_caster/ray_caster.py`

---

## Quick decision table

| Goal                                                                          | Helper                                                  |
| ----------------------------------------------------------------------------- | ------------------------------------------------------- |
| Validate that the plan covers a path                                          | `source_prim(plan, expr)[0] is None` (already in `AssetBase` / `SensorBase`) |
| Build a backend view glob that matches one prim per env                       | `descend_source_prims` (Pattern A)                      |
| Find the rigid-body ancestor of a sensor frame                                | `ascend_source_prims` (Pattern B)                       |
| Iterate per env inside an asset/sensor                                        | `destination.format(env_id)` in a loop (Pattern C default) |
| Hand a flat `list[str]` to a non-plan-aware downstream API                    | `expand_clone_plan_paths` (Pattern C boundary adapter)  |
| Compose world pose for a non-physics frame                                    | `source_prim` + `env_pose` + `combine_frame_transforms` (Pattern D) |
| Iterate raw plan matches                                                      | `iter_clone_plan_matches` (Pattern E)                   |

## Anti-patterns to remove

If you see any of these in a backend `_initialize_impl`, port them to one of
the patterns above:

- `find_first_matching_prim(self.cfg.prim_path)` followed by
  `get_all_matching_child_prims(destination, predicate=...)` — replace with
  Pattern A.
- `path.replace("env_0", "env_*")` or `path.replace("env_0", "env_.*")` —
  always wrong under heterogeneous clones; replace with `destination` from
  Pattern A or B.
- `re.sub(r"(?<=[Ee]nv_)\.\*", str(inst_idx), path)` — replace with
  `destination.format(inst_idx)` (Pattern C default). Reach for
  `expand_clone_plan_paths` only when the consumer needs a flat `list[str]`.
- Re-emitting `"<X> is not covered by the active ClonePlan"` in a subclass
  — the base classes already do this.
- Mixing `sim_utils.get_first_matching_ancestor_prim` with a plan lookup when
  you already have an `ascend_source_prims` shape — the canonical helper
  returns the ancestor *and* the `(source_root, destination)` you'd need
  to build the view glob, in one call.
