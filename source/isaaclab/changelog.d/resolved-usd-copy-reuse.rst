Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.assets.retrieve_file_path` leaving a new directory of resolved USD copies in the
  download directory on every call. The copies are now stored under a hash of each layer's source, contents and
  resolved references, written once, and reused when an unchanged tree is retrieved again. A benchmarking host had
  accumulated about 13,000 such directories a day (347 GB in eight days).
