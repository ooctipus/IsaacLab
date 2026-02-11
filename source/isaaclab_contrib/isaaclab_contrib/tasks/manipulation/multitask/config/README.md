# Multitask Design

This document describes how **multi-task environments register multiple task groups at runtime** from configuration under `multitask/config`, and how **assets, embodiments (robots), actions, sensors** are registered and adapted into **PerEnv\*** components.

---

## 1. Config entry: `MultiTaskRegistryConfig`

Multi-task behaviour is driven by `MultiTaskRegistryConfig` in `multitask/multitask.py`:

```python
@configclass
class MultiTaskRegistryConfig:
    task_names_by_group: list[str] = MISSING   # One Gym-registered task name per group
    group_size: int = 1                        # Number of envs per group
    device: str = "cuda:0"
    use_fabric: bool | None = None
```

- **`task_names_by_group`**: List length = number of groups; `task_names_by_group[group_idx]` is the **Gym-registered task name** for that group (e.g. `"Isaac-Stack-Cube-Franka-v0"`).
- **`group_size`**: Number of environments per task.
  - Mapping from env to group: `group_idx = env_idx // group_size`.
  - Env indices for group `group_idx` are `[group_idx * group_size, ..., group_idx * group_size + group_size - 1]` (from `env_indices_for_group(group_idx)`).

So **“runtime register multiple task_names_by_group”** means: when building the multi-task env config you pass a `MultiTaskRegistryConfig` whose `task_names_by_group` lists the task names to use per group. At runtime we do **not** register new tasks with Gym; we **resolve and cache** each task’s `ManagerBasedRLEnvCfg` by name.

---

## 2. Loading and caching task configs

- **Resolving**: `MultiTaskRegistryConfig.get_task_cfg(task_name)` calls `parse_env_cfg(task_name, ...)` (from `isaaclab_tasks.utils.parse_cfg`), which uses the Gym `env_cfg_entry_point` to load that task’s default env config (`ManagerBasedRLEnvCfg`).
- **Caching**: Each `task_name` is resolved once; the result is stored in `_task_cfg_cache` and reused for the same task in any group.

So **“how to register multi-task”** in practice is: in the **multi-task env’s `__post_init__`**, for each `group_idx` call `tasks.get_task_name_for_group(group_idx)` to get `task_name`, then `tasks.get_task_cfg(task_name)` to get that task’s scene/actions/rewards, and then register **assets, embodiments, actions, sensors** onto the current multi-task env config as described below.

---

## 3. Multi-task env config `__post_init__` flow

For `FrankaMultiTaskEnvCfg` (single-robot multi-task) and `MultiRobotMultiTaskEnvCfg` (multi-robot multi-task), `__post_init__` does the following in order:

1. **Validation**: `tasks` must be a `MultiTaskRegistryConfig` with `task_names_by_group` set.
2. **Shared ground and height alignment**: `_setup_shared_ground_plane_and_offsets()`
   - Uses the “first task that has a plane” to define a reference ground height.
   - Computes an `offset_z` per group; all assets in that group have `init_state.pos[2]` adjusted by this offset so different tasks’ ground/table heights align in one physical world.
3. **Robots (embodiments)**
   - **Single-robot** (`FrankaMultiTaskEnvCfg`): `_setup_robots()` keeps a single shared `scene.robot` but fills `_robot_init_state_tensor` per env (including per-group offset_z); on reset, `reset_multitask_robot_init_state` writes back per env.
   - **Multi-robot** (`MultiRobotMultiTaskEnvCfg`): `_setup_group_robots()` creates one `robot_group_{group_idx}` per group using **PerEnvArticulation**, with `prim_path` and `assigned_envs` as below.
4. **Actions**
   - **Single-robot**: `_setup_actions()` takes the actions config from the **first task that has actions** and uses it for all envs.
   - **Multi-robot**: `_setup_group_actions()` takes `arm_action` / `gripper_action` from each task’s `actions` per group, clones and points them at `robot_group_{group_idx}` (and that group’s gripper asset), and switches to **PerEnv\*** actions (see below).
5. **Register assets per group**: `_setup_group_assets()`.
6. **Register contact sensors per group**: `_setup_group_contact_sensors()` (may be commented out in some configs).
7. **Events / terminations / observations / rewards**:
   `_setup_group_events`, `_setup_group_terminations`, `_setup_group_observations`, `_setup_group_rewards` are currently mostly stubs or partial (TODO); the intended logic is to clone from each task_cfg and rewrite `SceneEntityCfg` to use group-named assets (e.g. `cube_1` → `cube_1_group_0`).

The following sections describe how **assets, embodiments, actions, sensors** are “registered” and adapted to PerEnv.

---

## 4. Multi-task assets: registration and PerEnv adaptation

In `_setup_group_assets()`, for each `group_idx`:

1. Get that group’s env indices: `env_tuple = tasks.env_indices_for_group(group_idx)`.
2. Get that group’s task config: `task_cfg = tasks.get_task_cfg(task_name)` with `task_name = tasks.get_task_name_for_group(group_idx)`.
3. Iterate over `task_cfg.scene` via `tasks.iter_scene_cfg_items(task_cfg.scene)`, skipping `robot`, `ee_frame`, `plane`, `light`.
4. For each asset config `asset_cfg`:
   - If `tasks.should_group_cfg(asset_cfg)` is False (i.e. `prim_path` has no `{ENV_REGEX_NS}` and does not start with `/World/envs/env_`), skip.
   - Otherwise treat the asset as per-env and register per group:
     - **prim_path**: Use `tasks.group_prim_from_template(env_tuple, asset_cfg.prim_path)` to turn the template path into a **regex matching only that group’s envs**, e.g.
       `/World/envs/env_{ENV_REGEX_NS}/...` → `/World/envs/env_(0|1|2|...|9)/...` when `env_tuple = (0..9)`.
     - **init_state**: If there is a ground height offset, use `_apply_ground_offset_to_init_state(asset_cfg.init_state, offset_z)`.
     - **class_type** (only for types that need PerEnv):
       - `ArticulationCfg` → `PerEnvArticulation`
       - `RigidObjectCfg` → `PerEnvRigidObject`
       - `SurfaceGripperCfg` → `PerEnvSurfaceGripper`
       - Plain `AssetBaseCfg` (e.g. table): only change `prim_path` and `init_state`, do not change class.
     - **assigned_envs**: Set `cloned.assigned_envs = env_tuple` on the cloned cfg.
5. Attach the asset to the scene: `setattr(self.scene, f"{asset_name}_group_{group_idx}", cloned)`.

So:
- **“How to register multi-task assets”**: Loop over groups, take assets from each group’s `task_cfg.scene`; for those passing `should_group_cfg`, clone, set `prim_path`, `init_state`, `assigned_envs`, set PerEnv* class type where applicable, and register on `scene` as `{asset_name}_group_{group_idx}`.
- **“How they become PerEnv”**: When cloning, pass `class_type=PerEnvArticulation` / `PerEnvRigidObject` / `PerEnvSurfaceGripper`. These classes inherit **PerEnvMixin**; at construction they get `assigned_envs` via `_resolve_assigned_envs(cfg)` (from cfg’s `assigned_envs` or by parsing `prim_path`), and in all `reset` / `write_*_to_sim` etc. they use `_filter_env_ids(env_ids)` so they only affect their assigned env subset.

---

## 5. Embodiments (robots): registration and PerEnv

- **Single-robot multi-task** (`FrankaMultiTaskEnvCfg`):
  - Registers a single `scene.robot` (from the first task that has a robot); it is not converted to PerEnv.
  - Uses `_robot_init_state_tensor[env_id]` and `reset_multitask_robot_init_state` to write different initial states per env on reset.

- **Multi-robot multi-task** (`MultiRobotMultiTaskEnvCfg`):
  - `_setup_group_robots()` takes `task_cfg.scene.robot` per group, clones and sets:
    - `prim_path = tasks.group_prim_from_template(env_tuple, robot_cfg.prim_path)`
    - `init_state = _apply_ground_offset_to_init_state(robot_cfg.init_state, offset_z)`
    - `class_type = PerEnvArticulation`
    - `cloned.assigned_envs = env_tuple`
  - Registers as `scene.robot_group_{group_idx}`.
  So each group’s robot is a **PerEnvArticulation** managing only that group’s envs, consistent with how other assets are adapted to PerEnv.

---

## 6. Actions: registration and PerEnv adaptation

- **Single-robot**: Actions are not split by group; the first task that has actions supplies the `actions` config for all envs. No PerEnv needed.

- **Multi-robot**: In `_setup_group_actions()`, for each group:
  - Take `arm_action`, `gripper_action`, etc. from `task_cfg.actions`.
  - Use `get_per_env_action_class(arm_action_src)` (from `per_env_action_factory`) to get the corresponding PerEnv action class if any.
  - Clone the action cfg and set:
    - `asset_name` to `robot_group_{group_idx}` (or that group’s gripper asset name);
    - If a PerEnv class exists: `class_type=per_env_class` and `arm_action.assigned_envs = env_tuple`.
  - Register as `self.actions.arm_action_group_{group_idx}`, etc.

**How PerEnv actions are created** (`per_env_action_factory.py`):
- `make_per_env_action(BaseAction, CfgClass)` dynamically builds a class that inherits from `PerEnvMixin` and the original action.
- In `__init__`, any buffer with `shape[0] == num_envs` (e.g. `_raw_actions`, `_processed_actions`) is sliced to `len(assigned_envs)`; internal `_ik_controller` / `_osc` may be wrapped with `make_per_env_controller_wrapper(assigned_envs, ctrl)`.
- In `reset(env_ids)`, `_filter_env_ids(env_ids)` is applied first, then `super().reset(resolved_env_ids)`.
- The registry `_CFG_TO_PER_ENV_CLASS` maps each ActionCfg type to its PerEnv class for `get_per_env_action_class(cfg)`.

---

## 7. Sensors (e.g. ContactSensor): registration and PerEnv

In `_setup_group_contact_sensors()`, for each group we iterate over `ContactSensorCfg` in `task_cfg.scene`; for those with `should_group_cfg(sensor_cfg)` True:
- Convert `prim_path` and `filter_prim_paths_expr` with `group_prim_from_template(env_tuple, ...)` to that group’s regex;
- Clone and set `class_type=PerEnvContactSensor`, `cloned.assigned_envs = env_tuple`;
- `setattr(self.scene, f"{sensor_name}_group_{group_idx}", cloned)`.

PerEnv sensors also use **PerEnvMixin** to resolve `assigned_envs` at construction and `_filter_env_ids` in their API so they only operate on their assigned envs.

---

## 8. Summary: from config to PerEnv components

| Config / component | Meaning | How it is registered in multi-task | PerEnv adaptation |
|-------------------|----------|--------------------------------------|--------------------|
| `task_names_by_group` | Gym task name per group | No new Gym registration; use `get_task_cfg(name)` to get existing task config | — |
| `group_size` | Envs per group | Used to compute `env_indices_for_group(group_idx)` | — |
| **Assets** (objects, table, gripper, etc.) | From each task’s scene | `_setup_group_assets()`: clone per group, set `prim_path` (regex), `init_state`, `assigned_envs`, attach as `scene.{name}_group_{idx}` | Articulation/RigidObject/SurfaceGripper → `PerEnvArticulation` / `PerEnvRigidObject` / `PerEnvSurfaceGripper` |
| **Embodiments** (robots) | Single-robot: one shared robot; multi-robot: one per group | Single: `_setup_robots()` one robot + per-env init_state tensor; multi: `_setup_group_robots()` one `robot_group_{idx}` per group | Multi-robot: robot uses `PerEnvArticulation` |
| **Actions** | Single: one shared set; multi: one set per group | Single: `_setup_actions()` takes first set; multi: `_setup_group_actions()` clones per group and binds to `robot_group_{idx}` | Multi: use `get_per_env_action_class(cfg)` for PerEnv class and set `assigned_envs`; internal controller wrapped with `PerEnvControllerWrapper` |
| **Sensors** (e.g. ContactSensor) | From task scene per group | `_setup_group_contact_sensors()`: clone per group, set `prim_path`, `assigned_envs` | `class_type=PerEnvContactSensor` |

All PerEnv components share:
- **Config**: must have **`assigned_envs`** (or an env set derivable from `prim_path`).
- **Implementation**: inherit **PerEnvMixin**, use `_filter_env_ids(env_ids)` to map global env_ids to “local indices for this component’s envs”, then call the base implementation so only the assigned env subset is affected.

---

## 9. PerEnvMixin: summary and adaptation table

**PerEnvMixin** (`utils/per_env_mixin.py`) provides per-environment scoping so a single component instance only operates on a subset of envs (e.g. one task group in a multi-task setup).

- **Resolution**: In `__init__`, `_resolve_assigned_envs(cfg)` sets `_assigned_envs` from the config’s `assigned_envs` or by parsing `prim_path` (e.g. `env_(0|1|2)` → `(0,1,2)`). Empty tuple means “all envs”.
- **Mapping**: `_assigned_env_to_local` maps global env index → local index for that component.
- **Filter**: `_filter_env_ids(env_ids)` normalizes and filters to the managed subset and returns **local indices** as a long tensor (or all local indices if `env_ids` is `None`).
- **Pattern**: Subclasses override methods that take `env_ids` (or `indices`): call `resolved = self._filter_env_ids(env_ids)`, early-return if empty, then `super().method(..., resolved)` so only the assigned env subset is affected. Optionally slice per-env tensor arguments (e.g. `states`, `positions`) to the same subset before calling super.

The following table lists the **base classes** that are adapted with PerEnvMixin, and for each: which **buffers** and **methods** need to be adapted (sliced or filtered).

| Base class | PerEnv class | Buffers to adapt | Methods to override (filter env_ids / indices) |
|------------|--------------|------------------|-----------------------------------------------|
| `RigidObject` | `PerEnvRigidObject` | None (base keeps full-size buffers; only env_ids are filtered when calling base) | `reset`, `write_root_pose_to_sim`, `write_root_velocity_to_sim` |
| `Articulation` | `PerEnvArticulation` | None (same as above) | `reset`, `write_root_pose_to_sim`, `write_root_velocity_to_sim`, `write_joint_state_to_sim`, `set_joint_position_target`, `set_joint_velocity_target` |
| `SurfaceGripper` | `PerEnvSurfaceGripper` | None | `reset`, `set_grippers_command`, `update_gripper_properties` (also slice `states` / per-env tensors to assigned subset before super) |
| `*ActionTerm` (e.g. `JointPositionAction`, `DifferentialInverseKinematicsAction`, `OperationalSpaceControllerAction`) | `PerEnv*Action` (via `make_per_env_action`) | All tensors with `shape[0] == num_envs` (e.g. `_raw_actions`, `_processed_actions`); slice to `len(assigned_envs)`. Wrap `_ik_controller` / `_osc` with `PerEnvControllerWrapper`. | `reset`; buffer slicing and controller wrapping in `__init__` |
| Controller (e.g. `DifferentialIKController`, `OperationalSpaceController`) | `PerEnvControllerWrapper` | All tensor attributes with `shape[0] == num_envs` or `num_robots` on the **wrapped** controller; set `num_envs` / `num_robots` to `len(assigned_envs)` | `reset`, `reset_idx` (filter env_ids / robot_ids then delegate) |
| `ContactSensor` | `PerEnvContactSensor` | None | `reset` |
| `Camera` | `PerEnvCamera` | None | `reset`, `set_intrinsic_matrices`, `set_world_poses`, `look_at` |
| `TiledCamera` | `PerEnvTiledCamera` | None | `reset`, `set_intrinsic_matrices`, and any other methods that take `env_ids` (same pattern as `PerEnvCamera`) |

When adding a new PerEnv variant for another base class:

1. Inherit from `PerEnvMixin` and the base class (order: `PerEnvMixin` first so `_filter_env_ids` is available).
2. In `__init__`, call `super().__init__(cfg, ...)` so that `PerEnvMixin.__init__` runs and sets `_assigned_envs` / `_assigned_env_to_local`.
3. Override every method that accepts `env_ids` (or `indices`): filter with `_filter_env_ids`, early-return if empty, then call `super().method(..., resolved_env_ids)`. If the method takes per-env tensors (e.g. `states`, `positions`), slice those to the same subset when `env_ids` is not None.
4. If the base allocates buffers with `shape[0] == num_envs`, either slice them in `__init__` to `len(assigned_envs)` (and treat indices as local), or ensure the base is only ever called with filtered env_ids and document that buffers remain full-size (current asset/sensor pattern).

---

## 10. File index

All files are located under `source/isaaclab_contrib/isaaclab_contrib/tasks/manipulation/multitask/`:

- **Multi-task config and helpers**
  - `multitask_cfg.py`: `MultiTaskRegistryConfig`, `iter_scene_cfg_items`, `clone_cfg`, `group_prim_from_template`, `should_group_cfg`
- **Multi-task env configs (this config package)**
  - `config/franka/franka_multitask_env_cfg.py`: single-robot multi-task (shared robot)
  - `config/franka/multirobot_multitask_env_cfg.py`: multi-robot multi-task (one robot + one action set per group)
- **PerEnv infrastructure**
  - `mixin_utils/per_env_mixin.py`: `PerEnvMixin`, `_resolve_assigned_envs`, `_filter_env_ids`
  - `mixin_utils/per_env_assets.py`: `PerEnvArticulation`, `PerEnvRigidObject`, `PerEnvSurfaceGripper`
  - `mixin_utils/per_env_action_factory.py`: `make_per_env_action`, `get_per_env_action_class`, PerEnv*Action classes
  - `mixin_utils/per_env_controller_factory.py`: `PerEnvControllerWrapper`, `make_per_env_controller_wrapper`
- **Multi-task specific modules**
  - `mdp/events.py`: Multi-task specific event functions
  - `mdp/__init__.py`: MDP module exports
- **Task config parsing**
  - Uses `isaaclab_tasks.utils.parse_cfg`: `parse_env_cfg(task_name, ...)`, resolution of Gym `env_cfg_entry_point`

---

## 11. Testing and Running

### Quick Start

To test the multi-task environment, use the random agent script:

```bash
# Single-robot multi-task (40 environments with shared robot)
python scripts/environments/random_agent.py --task Isaac-Franka-Multi-Task-v0 --num_envs 40

# Multi-robot multi-task (40 environments, different robots per group)
python scripts/environments/random_agent.py --task Isaac-MultiRobot-Multi-Task-v0 --num_envs 40 --device cpu
```

### Task Registration

Tasks are registered in `source/isaaclab_contrib/isaaclab_contrib/tasks/manipulation/multitask/config/franka/__init__.py`:

- `Isaac-Franka-Multi-Task-v0`: Single-robot multi-task environment
- `Isaac-MultiRobot-Multi-Task-v0`: Multi-robot multi-task environment


For higher-level architecture and data flow, see this document.
