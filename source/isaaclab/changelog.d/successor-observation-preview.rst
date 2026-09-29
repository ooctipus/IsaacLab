Added
^^^^^

* Added successor observation preview with current-sample history and delay, isolated
  built-in filters and noise state, and preserved Torch RNG. Preview required pure
  observation functions and supported built-in postprocessors; unsupported stateful
  callbacks were rejected before evaluation. Ordinary observation computation was unchanged.
