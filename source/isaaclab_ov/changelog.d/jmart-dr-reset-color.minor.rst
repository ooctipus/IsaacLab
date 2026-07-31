Added
^^^^^

* Added shared visual-material color sync to the OVRTX renderer: colors written to scene-level
  materials (which live outside the cloned environment namespace and are therefore shared by all
  environment clones) are mirrored into the detached native stage and temporal accumulation is
  invalidated.
