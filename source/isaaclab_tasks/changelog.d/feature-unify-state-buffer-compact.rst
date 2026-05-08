Added
^^^^^

* Promoted :meth:`~isaaclab_tasks.manager_based.multi_task.curriculum.StateBuffer.compact` from a private auto-trigger helper to a public idempotent method. Returns the surviving slot indices (same tensor passed to registered compact callbacks); is a no-op when the buffer already fits within ``target_size``. Lets one-shot callers (e.g. the locomotion task-table builder) trigger thinning explicitly without first overflowing to ``max_size``.
