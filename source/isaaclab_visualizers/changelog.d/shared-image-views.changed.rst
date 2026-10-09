* Separated shared image selection from Newton window dimensions through ``WindowCfg``.
  Recorders reused the view's completed frame. RTX capture copied native output directly on the GPU;
  only CPU consumers downloaded pixels. Sensor-view window resizing retained source and output storage.
