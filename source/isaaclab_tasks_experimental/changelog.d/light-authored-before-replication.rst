Fixed
^^^^^

* Fixed the Warp direct environments authoring ``/World/Light`` after their environments were
  replicated, which left the light out of the scene a renderer builds for itself from the stage and
  rendered those tasks unlit. The light is now authored with the rest of the scene, before
  replication, in the Cartpole, Reorient and locomotion Warp environments.
