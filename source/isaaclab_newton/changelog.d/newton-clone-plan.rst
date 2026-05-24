Changed
^^^^^^^

* Changed Newton ray-caster site registration to use
  :func:`~isaaclab.cloner.cloner_utils.iter_clone_plan_matches` instead of the
  destination-prefix string heuristic; per-env mesh counts now respect the plan
  rather than the ``site_count // num_envs`` assumption.
