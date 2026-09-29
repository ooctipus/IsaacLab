Changed
^^^^^^^

* Scheduled independent exact population construction and native state transfers on
  the Newton backend's existing CUDA streams. Added pinned transfer-status readback
  and completion tracking across caller streams while preserving atomic replacement,
  unchanged population reuse, and safe retirement.
