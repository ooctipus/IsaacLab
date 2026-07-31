Added
^^^^^

* Added per-environment :class:`~isaaclab.assets.VisualMaterial` entities: a
  ``{ENV_REGEX_NS}``-prefixed :attr:`~isaaclab.assets.VisualMaterialCfg.prim_path` is clone-planned
  and replicated like any other env-scoped spawn, with geometry binding its own environment's
  clone through the same token in ``visual_material_path`` / ``visual_material_bindings``. The
  whole-environment clone re-anchors these bindings to each environment through ``Sdf.CopySpec``
  (Kit renderers) and the Newton model's per-shape material map (Newton renderer). The detached
  OVRTX renderer does not yet remap cloned binding targets, so per-environment materials are not
  supported there.

Changed
^^^^^^^

* Changed :class:`~isaaclab.envs.mdp.randomize_visual_color` and
  :class:`~isaaclab.envs.mdp.randomize_visual_material` to follow the granularity the material
  entity declares: bucket materials keep the global write (``env_ids`` ignored), while
  per-environment materials sample one value per environment and honor ``env_ids``.
* Added ``omni.physx.fabric`` to the ``isaaclab.python.kit`` app manifest, matching the headless
  app's explicit dependency (source builds whose ``omni.physx.bundle`` does not resolve it need
  the extension for the PhysX fabric interface).
