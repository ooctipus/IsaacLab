Changed
^^^^^^^

* Added procedural keyboards with every multiple of six active keys from six to 108, retaining backspace.
* Registered keyboard meshes and physical properties before play; reset-time swaps updated mass, center of
  mass, inertia, joint properties, collision bounds, visibility, and episode participation together.
* Made reset-curriculum snapshots variant-specific and restored only snapshots compatible with the current keyboard.
* **Breaking:** Changed SO101 to 18 six-DOF keyboard partitions and native MJWarp contacts with sleeping.
  Custom selectors must use ``Keyboard/parts/part_*/keys`` and ``Keyboard/parts/part_*/joints`` paths.
  Case geometry moved onto the first partition's fixed root, and trim names became stable numbered slots.
  Set ``keyboard_variants=()`` for the all-active 108-key partitioned baseline. Otherwise,
  ``env.reset_keyboard(env_ids, variant_ids)`` selects registered variants for individual episode resets.
  ``TYPING_KEYBOARD_VARIANTS`` replaced the old ``TYPING_KEYBOARD_POOL``; generate metadata from its configurations.
