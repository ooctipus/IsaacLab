
Spawning Multiple Assets
========================

.. currentmodule:: isaaclab

Typical spawning configurations (introduced in the :ref:`tutorial-spawn-prims` tutorial) copy the same
asset (or USD primitive) across the different resolved prim paths from the expressions.
For instance, if the user specifies to spawn the asset at "/World/Table\_.*/Object", the same
asset is created at the paths "/World/Table_0/Object", "/World/Table_1/Object" and so on.

However, we also support multi-asset spawning with two mechanisms:

1. Rigid object collections. This allows the user to spawn multiple rigid objects in each environment and access/modify
   them with a unified API, improving performance.

2. Spawning different assets under the same prim path. This allows the user to create diverse simulations, where each
   environment has a different asset.

This guide describes how to use these two mechanisms.

The sample script ``multi_asset.py`` is used as a reference, located in the
``IsaacLab/scripts/demos`` directory.

.. dropdown:: Code for multi_asset.py
   :icon: code

   .. literalinclude:: ../../../scripts/demos/multi_asset.py
      :language: python
      :emphasize-lines: 100-118, 120-139, 141-174
      :linenos:

This script creates multiple environments, where each environment has:

* a rigid object collection containing a cylinder, a cube, and a sphere
* a rigid object distributed among cylinder, cube, and sphere variants
* an articulation distributed between the ANYmal-C and ANYmal-D variants

.. image:: ../_static/demos/multi_asset.jpg
  :width: 100%
  :alt: result of multi_asset.py


Rigid Object Collections
------------------------

Multiple rigid objects can be spawned in each environment and accessed/modified with a unified ``(env_ids, obj_ids)`` API.
While the user could also create multiple rigid objects by spawning them individually, the API is more user-friendly and
more efficient since it uses a single physics view under the hood to handle all the objects.

.. literalinclude:: ../../../scripts/demos/multi_asset.py
   :language: python
   :lines: 120-139
   :dedent:

The configuration :class:`~assets.RigidObjectCollectionCfg` creates the collection. Its
:attr:`~assets.RigidObjectCollectionCfg.rigid_objects` attribute is a dictionary containing
:class:`~assets.RigidObjectCfg` objects. The keys uniquely identify each object in the collection.


Spawning different assets under the same prim path
--------------------------------------------------

It is possible to spawn different assets and USDs under the same prim path in each environment using the spawners
:class:`~sim.spawners.wrappers.MultiAssetSpawnerCfg` and :class:`~sim.spawners.wrappers.MultiUsdFileCfg`:

* We set the spawn configuration in :class:`~assets.RigidObjectCfg` to be
  :class:`~sim.spawners.wrappers.MultiAssetSpawnerCfg`:

  .. literalinclude:: ../../../scripts/demos/multi_asset.py
     :language: python
     :lines: 100-118
     :dedent:

  This function declares the list of rigid-object variants. The clone plan assigns one
  variant to each environment and gives the spawner its exact prototype paths.

* Similarly, we set the spawn configuration in :class:`~assets.ArticulationCfg` to be
  :class:`~sim.spawners.wrappers.MultiUsdFileCfg`:

  .. literalinclude:: ../../../scripts/demos/multi_asset.py
     :language: python
     :lines: 141-174
     :dedent:

  Similar to before, this configuration allows the selection of different USD files representing articulated assets.


Similar asset structuring
~~~~~~~~~~~~~~~~~~~~~~~~~

While spawning and handling multiple assets using the same physics interface (the rigid object or articulation classes),
it is essential to have the assets at all the prim locations follow a similar structure. In case of an articulation,
this means that they all must have the same number of links and joints, the same number of collision bodies and
the same names for them. If that is not the case, the physics parsing of the prims can get affected and fail.

The main purpose of this functionality is to distribute variants of the same asset,
for example robots with different link lengths, or rigid objects with different collider shapes.

Physics replication in interactive scene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``scene.clone_cfg.replicate_physics`` is enabled by default. The flat clone plan gives the
physics backend a separate row for every active variant, so heterogeneous assets do not require
disabling physics replication. The example keeps it enabled.

Disable native physics replication only when a selected backend or prim type must parse each
environment from USD. Mandatory model-building contexts still consume the plan. Newton views
currently require a uniform body layout, so the example selects one robot and rigid-object variant
under Newton. For clone combinations and collision filtering, see :doc:`cloning`.

.. literalinclude:: ../../../scripts/demos/multi_asset.py
   :language: python
   :lines: 247-251
   :dedent:

The Code Execution
------------------

To execute the script with multiple environments and distributed asset variants, use the following command:

.. code-block:: bash

   uv run --extra isaacsim python scripts/demos/multi_asset.py --num_envs 2048

This command runs the simulation with 2048 environments populated from the declared asset variants.
To stop the simulation, you can close the window, or press ``Ctrl+C`` in the terminal.
