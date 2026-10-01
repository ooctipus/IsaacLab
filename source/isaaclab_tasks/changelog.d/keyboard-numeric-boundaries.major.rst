Changed
^^^^^^^

* Separated keyboard path resolution from integer selection binding. Moved
  ``NewtonSelectorCfg`` and symbolic preparation into ``selection_paths``; replaced
  selection-owner ``resolve`` with integer-only ``bind``. Bound selections retained
  no path configuration; environment configurations remained declarative while
  managers owned resolved copies. Distinguished static entity counts from padded
  policy width and shared one immutable native conversion map between selections
  and reset payloads.
