Added
^^^^^

* Added shared visual-material color sync to the kit-less Newton-Warp renderer: it groups
  ``model.shape_label`` rows by the USD material each shape binds (resolving cloned environments
  through the published clone plan), converts linear RGB to sRGB, and scatters the notified bucket
  colors through a zero-copy Torch view of ``model.shape_color``.
