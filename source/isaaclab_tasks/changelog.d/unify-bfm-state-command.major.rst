Changed
^^^^^^^

* **Breaking:** Rebuilt the multi-task ``StateCommand`` stack on composable task-table
  families. ``StateCommandCfg.TaskTableCfg`` now declares ``families`` of generate, solve,
  criteria, and selection stages plus a table ``seed``, and builders run as pure functions
  ``build(command_cfg, scene_cfg, device)`` instead of reading a live environment. Position
  and Factory command tables, payloads, and curriculum bindings were migrated; payloads now
  own reset writes through :class:`ResetStateBank` / :class:`ResetStateWriter`, and the
  curriculum term owns ``success_rates`` (bind ``coords_bind`` to
  ``table.states.root_pose`` and drop ``success_rates_bind``).
* **Breaking:** Removed the monolithic terrain ``RetargetPipeline`` / ``RetargetPipelineCfg``
  and the factory retarget pipeline. Declare the sampler on
  ``PositionTerrainStanceGenerateCfg``, IK terms as ``IKObjective*Cfg`` entries of
  ``PositionIKSolveCfg.objectives`` (matched to generated targets through ``target_bind``),
  acceptance criteria on the family, and FPS thinning on ``PositionFpsSelectionCfg``.
  ``IKObjectiveTerrainCollisionCfg`` / ``IKObjectiveTerrainContactCfg`` were replaced by the
  mesh-generic ``IKObjectiveMeshCollisionCfg``.
* **Breaking:** Rebuilt :class:`NewtonKinematics` around a shared, validated ``Topology``
  (USD or MJCF, ``from_articulation`` for scene assets), memory-bounded batched IK
  execution (``execute_ik_batches``) on the continuous Newton ``IKSolver.solve`` API, and
  objectives built from explicit :class:`IKObjectiveBuildContext` data instead of a pipeline.
* Made preset resolution in :mod:`isaaclab_tasks.utils.hydra` traverse tuples and copy
  selected preset values so nested :class:`PresetCfg` instances inside tuple fields resolve
  without mutating the shared templates.
* Added an optional ``generator`` argument to
  :func:`isaaclab_tasks.core.multi_task.utils.grid_downsample.grid_bucket_downsample` and
  to the morphological flat-patch sampler so table construction consumes one explicit
  random stream.
* Extended the optional Position ``PhysicalSuccessGateCfg`` with per-body linear and
  angular settle-speed ceilings applied while accumulating successful hold time.
