Added
^^^^^

* Added :meth:`~isaaclab_ov.cloner.OvReplicateContext.create_ovstage`, which builds a populated OVStage
  stage from the clone-plan snapshot so a renderer such as Newton's ``ViewerRTX`` can draw it with
  per-environment MDL materials intact. A consumer declares itself with
  ``OvReplicateContext._request_ovstage`` before the clone plan completes.
* Added a ``gpu_hierarchy`` argument to :func:`isaaclab_ov.stage.create_ovstage`. A stage borrowed by
  ``ViewerRTX`` needs GPU hierarchy computation, which OVStage cannot combine with the CPU hierarchy
  stage OVPhysX keeps alive in the same process.
