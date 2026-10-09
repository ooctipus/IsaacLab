* Added shared ``ImageViewCfg`` selections for scene cameras and perspective renders. Windows and
  ``VideoRecorderCfg(view=...)`` reused one composed device image and one host readback per frame.
  Kept legacy source strings and camera configurations; new configurations use
  ``NewtonGLVisualizerCfg(window=WindowCfg(view=view))`` or the RTX equivalent.
* Consolidated display and recording colorization in the device composition kernels, removing the
  duplicate CPU implementation and obsolete camera colorization, gathering, and grid helpers.
