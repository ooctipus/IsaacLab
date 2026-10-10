* Separated shared image selection from Newton window dimensions through ``WindowCfg``.
  Recorders reused the view's completed frame. RTX capture copied native output directly on the GPU;
  only CPU consumers downloaded pixels. Sensor-view window resizing retained source and output storage.
  ``WindowCfg.size`` set initial viewer dimensions; RTX perspective rendering kept that resolution,
  while GL perspective rendering followed its window framebuffer.
  Unified GL and RTX capture lifecycle checks; call ``render_rgb_array()`` after ``sim.reset()``.
