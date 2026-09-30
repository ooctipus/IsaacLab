Scene Data Provider
===================

The :class:`~isaaclab.scene_data.SceneDataProvider` is the data boundary between physics and
rendering. Physics publishes native pointers, their formats, and dirty latches. A renderer or
visualizer requests the format it consumes. Neither side imports or calls the other.

Data flow
---------

A :class:`~isaaclab.scene_data.SceneDataBackend` publishes:

* one ``transform_publication``;
* named ``point_publications``.

Every publication is a :class:`~isaaclab.scene_data.SceneDataPublication` containing only its
native-format pointer bundle and dirty latch. Flat formats such as
:class:`~isaaclab.scene_data.SceneDataFormat.Points` obtain their count from the pointer's leading
dimension. Native padded-body and cable bundles do not publish a second count; their count and
ordering come from the bound :class:`~isaaclab.cloner.ClonePlan`.

Two native point formats avoid eager physics-side packing:

* :class:`~isaaclab.scene_data.SceneDataFormat.BodyPoints` contains padded deformable-body pointers
  and the plan binding IDs for their rows. PhysX publishes its native nodal arrays directly;
  OVPhysX may read into one native staging array when its API requires a destination.
* :class:`~isaaclab.scene_data.SceneDataFormat.CablePoints` contains Newton body/model pointers and
  plan-derived cable segment topology. Cable endpoints do not exist as a second physics buffer;
  SDP derives them only when a consumer requests a point destination.

The provider exposes two request operations:

.. code-block:: python

   transforms = provider.request_transforms(SceneDataFormat.Transform)
   particles = provider.request_points(SceneDataFormat.Points, "points")

When the requested format is native, the returned object aliases the published pointer. No kernel
runs and no buffer is allocated. For native point bundles, SDP first validates their pointer and
binding cardinality against the bound layout. Otherwise, the provider allocates the requested
output and performs one fused gather, interpolation, coordinate conversion, or host transfer for
the current dirty generation.
Repeated requests for the same format and generation return that cached output.
Physics materializes each publication at its reset, forward, or step boundary. An SDP request only
reads the published pointer and dirty latch; it never invokes physics or deferred producer work.

Fabric is another requested format. Kit and Isaac RTX request
:class:`SceneDataFormat.FabricMatrix44` and :class:`SceneDataFormat.FabricMeshPoints`; the provider
writes the selected Fabric arrays. The shared USD clone context materializes those destination
pointers from the completed plan; SDP never attaches, searches, or selects a stage. Cable,
deformable, and MPM destinations use the same named point-publication contract rather than
backend-specific synchronization callbacks.
OVRTX requests :class:`SceneDataFormat.TransposedMatrix44d`, its exact ``omni:xform`` sink layout,
so its renderer never gathers, casts, or transposes published physics transforms.

Clone-plan topology
-------------------

Publications do not carry paths, counts, or destination mappings. Physics publishes transforms in
the clone plan's canonical rigid-body order. A native point bundle carries only the binding IDs and
source topology needed to interpret its native pointers. Those values are derived from the plan;
the provider binds the exact completed :class:`~isaaclab.cloner.ClonePlan` once and materializes
its static counts and point maps. Frame and geometry facts remain one record per prototype with a read-only
clone-column mask; queries materialize only their selected exact destinations. Full frame expansion
is reserved for USD/Fabric setup. Consumers therefore obtain transform destinations and point-stream
bindings before initialization rather than rediscovering geometry by walking the stage.

Environment count comes from the active clone plan. Renderer-visible scene composition must also
come from that plan; a stage walk is not a substitute for missing clone-plan coverage.

Backend resources
-----------------

The native backend registry on :class:`~isaaclab.sim.SimulationContext` is orthogonal to SDP.
Consumers derive a stable key from their configs and get or create the corresponding native
resource. Matching Newton physics and rendering configs therefore share one Newton model and
state naturally. Dynamic renderer input still passes through an SDP native-format request, which
returns the same pointer with zero conversion.

For a cross-backend renderer, its native rendering resource is built from the clone plan and its
dynamic transform or point pointers are populated from SDP in the requested format. There is no
renderer-to-physics dependency and no fallback stage synchronization path.

Lifecycle
---------

Physics, renderers, and visualizers are constructed explicitly from their configs. They register
their native resources before cloning, consume the same clone plan during one replication
lifecycle, and initialize runtime resources only after cloning completes. Each visualizer receives
the exact shared provider and clone plan at initialization; it does not recover topology from SDP.

See Also
--------

- :doc:`/source/overview/core-concepts/renderers`: renderer backends that consume scene data
- :doc:`/source/concepts/visualization`: visualizer backends that consume scene data
