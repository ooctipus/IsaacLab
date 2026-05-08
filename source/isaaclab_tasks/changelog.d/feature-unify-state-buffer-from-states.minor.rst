Added
^^^^^

* Added :meth:`~isaaclab_tasks.manager_based.multi_task.curriculum.StateBuffer.from_states` -- a view-wrap classmethod that constructs a one-shot :class:`StateBuffer` over a caller-owned state slab without allocating its own ``[max_size, state_dim]`` storage. ``compact()`` in this mode allocates a fresh ``[target_size, state_dim]`` output for the survivors and replaces ``self.data``, leaving the caller's input untouched. Streaming construction (``__init__``) keeps its existing in-place compaction semantics; a private ``_owns_data`` flag distinguishes the two modes inside ``compact()``. With this addition, ``StateBuffer`` is the canonical state-holder for both lifecycles -- streaming via ``add()`` + auto-compact, one-shot via ``from_states()`` + explicit ``compact()``.

Changed
^^^^^^^

* :func:`~isaaclab_tasks.manager_based.multi_task.terrain.retarget.apply_final_fps` now delegates the custom-extractor / spacing-driven path to :meth:`~isaaclab_tasks.manager_based.multi_task.curriculum.StateBuffer.from_states`. The xyz-default fast-path stays inline (it skips the full-slab gather, saving ~88 MB of transient memory at ``N=1M``) -- the function is now exactly "the in-place RetargetBuffer-flavoured entry point of ``from_states + compact``". No behaviour change for callers; pure abstraction reorganisation.
