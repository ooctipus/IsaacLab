# Multitask Config

This document describes the **Isaac Lab core API changes** that support multitask (heterogeneous) setups, how the **multitask config** uses them, and how to **test** with `random_agent.py` and what to watch out for.

---

## 1. Core API Modifications and Enhancements for Multitask

The following additions and changes in Isaac Lab core enable one simulation to run multiple task groups, each with its own subset of environments, assets, and actions.

### 1.1 Assigned environment indices

**Assets (Articulation, RigidObject, etc.)**

- **Config**: Asset configs may specify **`assigned_envs`** (a tuple of global env indices). If omitted, assignment is inferred from **`prim_path`** (or `spawn.prim_path`).
- **Resolution**: `isaaclab.utils.string.resolve_assigned_env_ids_from_cfg(cfg)`:
  - If `cfg.assigned_envs` is set, that tuple is used.
  - Otherwise, `extract_env_ids_from_prim_path(prim_path)` parses the path:
    - **`env_(0|1|2)`** or **`env_0`**, **`env_1`** → tuple `(0, 1, 2)` (this asset manages only those envs).
    - **`env_.*`** or **`{ENV_REGEX_NS}`** → treated as “all envs”; resolved assignment is an **empty tuple**.
  - Empty tuple means the asset manages all environments (homogeneous behaviour).
- **Runtime**: In `isaaclab.assets.articulation.base_articulation.Articulation` and `isaaclab.assets.rigid_object.base_rigid_object.RigidObject`, `__init__` sets:
  - `_assigned_envs`
  - `_assigned_envs_to_local_indices` (global env id → local index).
- **Properties**: `assigned_envs`, `assigned_envs_to_local_indices`, `is_heterogeneous` (true when `len(_assigned_envs) > 0`).
- **Filtering**: `_filter_env_ids(env_ids)` maps global env ids to local indices for the managed subset (or returns all local indices when `env_ids is None`).

**Managers (e.g. action terms)**

- **ManagerTermBase** (`isaaclab.managers.manager_base`): Adds `_assigned_envs`, `_assigned_envs_to_local_indices`, `assigned_envs`, `assigned_envs_to_local_indices`, `is_heterogeneous`, and `_filter_env_ids`. When `_assigned_envs` is non-empty, `num_envs` is `len(_assigned_envs)`.
- **ActionManager** / **ActionTerm**: After construction, the action term’s **`_assigned_envs`** is taken from the **asset** it references (`self._asset.assigned_envs`). So each action term automatically scopes to the same env subset as its asset.
- **process_actions**: In `ActionManager.process_actions(action)`, when a term has non-empty `assigned_envs`, the slice of `action` passed to that term is **`action[term_assigned_envs, idx : idx + term.action_dim]`**, so only the relevant envs receive that term’s actions.

### 1.2 Automatic env_ids filtering in asset methods

- **Decorator**: `isaaclab.utils.decorators.filter_env_ids_arg` wraps methods that take an `env_ids` argument. When `self.is_heterogeneous` is true, it replaces `env_ids` with `self._filter_env_ids(env_ids)` before calling the original method.
- **Usage**: Applied in `Articulation` and `RigidObject` via `__init_subclass__`: all subclass methods that accept `env_ids` are wrapped (except those in `FilterEnvIdsSkipMethodNames`), so reset/write_*_to_sim etc. automatically operate on local indices when the asset is heterogeneous.

### 1.3 Prim path parsing

- **`extract_env_ids_from_prim_path(prim_path)`** (`isaaclab.utils.string`): Parses `env_<id>` and `env_(a|b|c)` from a path; returns a sorted tuple of env indices, or `None` for generic “all envs” patterns (`env_.*`, `{ENV_REGEX_NS}`).
- **`resolve_assigned_env_ids_from_cfg(cfg)`**: Uses `assigned_envs` if present, else `prim_path` (or `spawn.prim_path`) via `extract_env_ids_from_prim_path`; returns empty tuple when the path means “all envs”.

These allow multitask configs to set **group-specific prim paths** (e.g. `/World/envs/env_(0|1|2)/Robot`) so that each asset instance is bound to a subset of envs without extra wrapper classes.

---

## 2. How the Multitask Config Uses the Core API

### 2.1 MultiTaskRegistryConfig

- **Location**: `multitask_utils.MultiTaskRegistryConfig`.
- **Role**: Drives which Gym task config is used per group and how envs are grouped.
- **Fields**: `task_names_by_group` (one Gym task name per group), `group_size` (envs per group), `device`, `use_fabric`.
- **Helpers**: `get_task_cfg(task_name)` loads and caches `ManagerBasedRLEnvCfg` via `parse_env_cfg`; `env_indices_for_group(group_idx)` returns the env indices for that group; `group_prim_from_template(env_tuple, prim_path)` builds a group-specific prim path regex (e.g. `/World/envs/env_(0|1|2)/...`); `should_group_cfg(cfg)` detects per-env configs (e.g. by `prim_path`).

### 2.2 Single-robot multitask (SingleRobotMultiTaskEnvCfg)

- One shared **robot** and one shared **actions** config (from the first task that has them). Robot is not scoped by env subset; per-env initial state is handled by `_robot_init_state_tensor` and a reset event (`reset_multitask_robot_init_state`).
- **Assets** per group: For each group, assets from that group’s task scene are cloned with:
  - **prim_path** = `group_prim_from_template(env_tuple, asset_cfg.prim_path)` (so core infers `assigned_envs` from the path),
  - **init_state** adjusted by ground offset.
  No change to asset class types; core Articulation/RigidObject handle scoping.

### 2.3 Multi-robot multitask (MultiRobotMultiTaskEnvCfg)

- One **robot** per group: `scene.robot_group_{group_idx}`, each with `prim_path` set via `group_prim_from_template` and optional ground offset. Core resolves `assigned_envs` from the path.
- **Actions** per group: For each group, action configs are cloned and **asset_name** is set to `robot_group_{group_idx}` (or that group’s gripper asset). Action terms then get `assigned_envs` from the asset; `ActionManager.process_actions` slices actions per term as described above.

So the multitask config only **composes and clones configs** and sets **prim_path** (and optionally **assigned_envs**); it does not introduce new asset or action classes. All scoping is done by the core API.

---

## 3. Testing with random_agent.py

### 3.1 Commands

From the repository root:

```bash
# Single-robot multitask (shared robot across envs), user cannot modify num_envs
python scripts/environments/random_agent.py --task Isaac-Franka-Multi-Task-v0

# Multi-robot multitask (one robot type per group), user cannot modify num_envs
python scripts/environments/random_agent.py --task Isaac-MultiRobot-Multi-Task-v0--device cpu
```

### 3.2 Task registration

Tasks are registered in **`config/demo/__init__.py`**:

- **Isaac-Franka-Multi-Task-v0**: Single-robot multitask (e.g. FrankaMultiTaskEnvCfg).
- **Isaac-MultiRobot-Multi-Task-v0**: Multi-robot multitask (e.g. MultiRobotMultiTaskManipulationEnvCfg).

### 3.3 Notes and caveats

- **num_envs and group_size**: `num_envs` should be consistent with the number of groups and `group_size` in your `MultiTaskRegistryConfig` (e.g. `num_envs = total_groups * group_size` for full batches). The demo configs use `group_size=10`; with 4 groups that implies 40 envs.
- **Device**: Multi-robot should be tested with `--device cpu` if SurfaceGripper is included in any of the tasks.
- **Observations / rewards / terminations**: The current multitask env configs use minimal proxy observations, rewards, and terminations; full per-task rewards and terminations may still be TODO and wired later with env_ids filtered by group.
- **Reset**: Single-robot multitask uses `reset_multitask_robot_init_state` to write per-env robot initial state from `_robot_init_state_tensor`; ensure events are registered so reset is called when expected.

---

## 4. File index

Under `source/isaaclab_contrib/isaaclab_contrib/tasks/manipulation/multitask/`:

- **multitask_utils.py**: `MultiTaskRegistryConfig`, `iter_scene_cfg_items`, `clone_cfg`, `group_prim_from_template`, `should_group_cfg`
- **multitask_env_cfg.py**: `SingleRobotMultiTaskEnvCfg`, `MultiRobotMultiTaskEnvCfg`, scene and setup helpers
- **config/demo/demo_multitask_env_cfg.py**: Example configs (`FrankaMultiTaskEnvCfg`, `MultiRobotMultiTaskManipulationEnvCfg`)
- **config/demo/__init__.py**: Gym task registration for the demo multitask envs
- **mdp/events.py**: Multitask-specific event functions (e.g. reset_multitask_robot_init_state, reset_multitask_scene_to_default)

Core API references (in main Isaac Lab source):

- **isaaclab.utils.string**: `resolve_assigned_env_ids_from_cfg`, `extract_env_ids_from_prim_path`
- **isaaclab.utils.decorators**: `filter_env_ids_arg`, `FilterEnvIdsSkipMethodNames`
- **isaaclab.assets** (base_articulation, base_rigid_object): `assigned_envs`, `_filter_env_ids`, heterogeneous handling
- **isaaclab.managers** (manager_base, action_manager): `assigned_envs` on terms, action slicing by `term_assigned_envs` in `process_actions`
