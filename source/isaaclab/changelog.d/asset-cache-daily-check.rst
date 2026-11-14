Changed
^^^^^^^

* Changed the local copies of remote assets to be checked against the server at most once a day: a copy whose revision
  the server confirmed within the last 24 hours is used without a query. The check cost a server round trip per file
  of each asset's dependency tree on every start-up, about 1.6 s for the G1 humanoid.
