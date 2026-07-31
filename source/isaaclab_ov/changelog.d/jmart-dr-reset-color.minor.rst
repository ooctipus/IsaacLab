Added
^^^^^

* Added shared visual-material color sync to the OVRTX renderer: colors written to scene-level
  materials (which live outside the cloned environment namespace and are therefore shared by all
  environment clones) are mirrored into the detached native stage and temporal accumulation is
  invalidated. Writes are unsupported on the opt-in ovstage path (``ISAAC_LAB_OVRTX_USE_OVSTAGE=1``),
  which rejects ``ovrtx_write_attribute`` in borrow mode; there the write is dropped with a one-time
  warning and the scene renders each material's authored color, so randomization is visibly absent
  rather than fatal. Unset the variable to use the legacy path, or randomize on the Newton /
  Isaac RTX backends.
