Changed
^^^^^^^

* Renamed task-local ``native_selection`` to ``mujoco_selection`` and its owners to
  ``MuJoCoSelection``, ``MuJoCoSelections``, ``EnvWorldBindings`` and
  ``NewtonMuJoCoMapping``. Replaced ``static_counts`` with ``world_selection_counts``
  or ``prototype_selection_counts``, and selector ``frequency``/``dense_width``
  with ``index_domain``/``policy_width``. Paths now resolve through
  ``query_selection_indices`` before numeric binding.
* Distinguished ``request_variants``, growable-world ``stage_variant_changes``,
  rebuilt-population ``apply_pending_variants`` and ``reset_from_snapshot`` operations.
  Growable worlds expose ``staged_variant_ids``; rebuilt populations expose
  ``committed_variant_ids``. Callers must select the operation matching their backend.

Fixed
^^^^^

* Rejected coordinate/DOF pairs with different owners, world placements or ordered
  scalar joints even when their selection counts matched.
* Validated grouped world coverage before publishing replacement selections, and
  updated staged variant IDs only after successful snapshot publication.
* Checked selected fields against the prepared model's index-domain schema and rejected
  invalid indexed writes before modifying state. Retired selections now reject cached
  and uncached access; captured kernels retain their exact buffers.
