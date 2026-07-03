Fixed
^^^^^

* Fixed the OVRTX renderer dropping the authored root scale of rigid objects: the per-body
  ``omni:xform`` written from physics poses replaced the prim's whole transform stack with a
  unit-scale matrix, so scaled rigid objects rendered at unit scale. Each bound body's authored
  local scale is now recomposed into the synced transform.
