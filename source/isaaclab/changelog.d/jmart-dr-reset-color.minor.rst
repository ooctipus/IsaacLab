Added
^^^^^

* Added :class:`~isaaclab.assets.VisualMaterial`, a scene-level material entity declared at one
  concrete prim path outside the environment namespace. Geometry in every environment can bind it
  through an absolute ``visual_material_path`` on the shape/mesh/USD-file spawners, so one material
  is shared by all environments on every rendering backend.
* Added :class:`~isaaclab.sim.spawners.materials.PbrMdlCfg` for spawning OmniPBR materials, and
  kit-less (pure ``UsdShade``) fallbacks for :func:`~isaaclab.sim.spawners.materials.spawn_preview_surface`,
  :func:`~isaaclab.sim.spawners.materials.spawn_from_mdl_file`, and
  :func:`~isaaclab.sim.utils.bind_visual_material`.
* Added :attr:`~isaaclab.sim.spawners.from_files.FileCfg.visual_material_bindings` to bind existing
  shared materials to specific parts of a file-spawned asset (asset-relative prim path → material
  prim path), enabling per-part color randomization granularity.

Changed
^^^^^^^

* **Breaking:** Reworked :class:`~isaaclab.envs.mdp.randomize_visual_color` around shared "bucket"
  materials: the term now targets a list of :class:`~isaaclab.assets.VisualMaterial` entities via
  ``materials`` and samples one color per material per fire, which recolors all environments that
  bind it on every renderer (Isaac RTX, OVRTX, Newton-Warp). The former per-environment parameters
  ``asset_cfg``, ``visual_path``, and ``sensor_cfg`` are removed, ``env_ids`` is ignored by design,
  and the ``replicate_physics=False`` requirement is lifted. Migration: declare bucket materials in
  the scene, bind assets to them with an absolute ``visual_material_path``, and pass
  ``materials=[SceneEntityCfg(...)]`` to the term; bind different environments to different buckets
  for cross-environment diversity.
