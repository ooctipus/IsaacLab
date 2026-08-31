Changed
^^^^^^^

* **Breaking:** Made the experimental manager-based and direct environment roots construct their
  scene cfg inside one ``ReplicateSession``. Custom environment roots must supply all clone-owned
  cfgs to that lifecycle before constructing assets.
