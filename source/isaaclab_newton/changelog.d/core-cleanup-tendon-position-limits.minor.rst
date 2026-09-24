Added
^^^^^

* Added fixed tendon position limits to :class:`~isaaclab_newton.assets.Articulation`:
  :meth:`~isaaclab_newton.assets.Articulation.set_fixed_tendon_position_limit_index` and
  :meth:`~isaaclab_newton.assets.Articulation.set_fixed_tendon_position_limit_mask` now write the MuJoCo tendon
  range instead of raising :class:`NotImplementedError`.

* Implemented fixed tendon limit-stiffness reads and index/mask setters through Newton's tendon force-gain
  attributes. Newton handled the solver conversion and refreshed it after inertia changes; zero stiffness
  disabled the limit and untouched tendons preserved their imported MuJoCo parameters.

Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.assets.Articulation.set_fixed_tendon_stiffness_mask` and
  :meth:`~isaaclab_newton.assets.Articulation.set_fixed_tendon_damping_mask` raising ``AttributeError``.
* Fixed :meth:`~isaaclab_newton.assets.Articulation.write_fixed_tendon_properties_to_sim_mask` raising
  ``TypeError``, and added the ``fixed_tendon_ids`` and ``fixed_tendon_mask`` arguments that the base class
  declares to the fixed tendon writers.
* Corrected fixed tendon errors and documentation to distinguish unimplemented Isaac Lab properties from
  Newton's MuJoCo tendon support.
* Fixed selected tendon property writes reading staged values from the first environment or tendon instead
  of the selected indices.
