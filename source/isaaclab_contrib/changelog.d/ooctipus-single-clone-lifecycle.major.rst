Changed
^^^^^^^

* **Breaking:** Made contributed deformable objects resolve their exact prototype paths from the
  active clone plan. Construct them inside the cfg-owned ``ReplicateSession`` lifecycle.
