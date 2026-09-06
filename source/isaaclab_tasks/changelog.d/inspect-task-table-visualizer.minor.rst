Added
^^^^^

* Added a ``--visualizer`` selector to the shared task-table inspector
  ``core/multi_task/scripts/inspect_task_table.py`` choosing between the ``viser`` web viewer
  (default) and the ``newton_gl`` OpenGL window. Static tables are now re-submitted at a low
  rate so windowed viewers keep processing input until closed.
* Added a ``--sequences`` count to the inspector and spread the displayed sequences evenly
  across the table instead of showing the first rows, so region-ordered tables such as the
  Position spawn/target pairs show every terrain cell rather than only the first one.
