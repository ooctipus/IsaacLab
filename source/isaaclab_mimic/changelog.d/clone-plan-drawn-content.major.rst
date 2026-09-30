Changed
^^^^^^^

* **Breaking:** Removed ``occupancy_map_add_to_stage`` and the locomanipulation data generator's
  ``--draw_visualization`` option. Runtime code may no longer author an unplanned mesh and material
  after cloning; declare any scene visualization through the root environment cfg instead.
