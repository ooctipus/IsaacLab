Changed
^^^^^^^

* Changed :meth:`~isaaclab.scene_data.SceneDataProvider.create_mapping` to check identical layouts for unique paths
  with one set instead of two index dictionaries, about 0.3 s for G1's 753K bodies at 16,384 environments.
