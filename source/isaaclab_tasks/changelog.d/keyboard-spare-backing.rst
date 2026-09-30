Changed
^^^^^^^

* Limited retained spare GPU backing for native keyboard worlds to 1 GiB during
  existing backing maintenance. Setting ``worlds_spare_memory_budget_bytes=None``
  preserved all available spare backing for subsequent growth.
