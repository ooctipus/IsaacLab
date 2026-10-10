* Added shared ``ImageViewCfg`` selections for scene cameras and perspective renders. Windows and
  ``VideoRecorderCfg(view=...)`` reused one composed device image and one host readback per frame.
  Kept legacy source strings and camera configurations; new configurations use
  ``NewtonGLVisualizerCfg(window=WindowCfg(view=view))`` or the RTX equivalent.
  Views retained native tile dimensions and selected existing channels; depth colorization left
  source measurements unchanged. Sensor resolution remained configured through ``CameraCfg``.
* Consolidated display and recording colorization in the device composition kernels, removing the
  duplicate CPU implementation and obsolete camera colorization, gathering, and grid helpers.
* Retained each configured perspective producer during headless recording, including multiple viewers
  of the same backend. Bound legacy sensor recordings directly to their scene sensor and stopped
  invalid image streams with one error instead of interrupting simulation.
