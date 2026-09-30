Changed
^^^^^^^

* **Breaking:** Changed CuRobo planning to construct its collision world exclusively from
  clone-plan geometry selected by ``CuroboPlannerCfg.mesh_prim_paths``. The task copies those
  expressions to ``InteractiveSceneCfg.geometry_prim_paths`` before scene construction and builds
  the planner with ``cfg.class_type(cfg, env, env_id)``. The planner no longer supports task-name
  factories, completed-stage discovery, or planner-owned Rerun/USD visualizers. Configure scene
  visualization through the simulation cfg instead.
