.. _tutorial-spawn-prims:


Spawning prims into the scene
=============================

.. currentmodule:: isaaclab

This tutorial explores how to spawn various objects (or prims) into the scene in Isaac Lab from Python.
It builds on the previous tutorial on running the simulator from a standalone script and
demonstrates how to spawn a ground plane, lights, primitive shapes, and meshes from USD files.

.. note::

   This tutorial automatically tetrahedralizes a volume deformable. Run it with the
   ``tetrahedralization`` extra:

   .. code-block:: bash

      uv run --extra tetrahedralization python scripts/tutorials/00_sim/spawn_prims.py

   With the legacy installer, install the optional dependencies first:

   .. code-block:: bash

      ./isaaclab.sh -i tetrahedralization


The Code
~~~~~~~~

The tutorial corresponds to the ``spawn_prims.py`` script in the ``scripts/tutorials/00_sim`` directory.
Let's take a look at the Python script:

.. dropdown:: Code for spawn_prims.py
   :icon: code

   .. literalinclude:: ../../../../scripts/tutorials/00_sim/spawn_prims.py
      :language: python
      :emphasize-lines: 40-88, 100-101
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Scene designing in Omniverse is built around a software system and file format called USD (Universal Scene Description).
It allows describing 3D scenes in a hierarchical manner, similar to a file system. Since USD is a comprehensive framework,
we recommend reading the `USD documentation`_ to learn more about it.

For completeness, we introduce the must know concepts of USD in this tutorial.

* **Primitives (Prims)**: These are the basic building blocks of a USD scene. They can be thought of as nodes in a scene
  graph. Each node can be a mesh, a light, a camera, or a transform. It can also be a group of other prims under it.
* **Attributes**: These are the properties of a prim. They can be thought of as key-value pairs. For example, a prim can
  have an attribute called ``color`` with a value of ``red``.
* **Relationships**: These are the connections between prims. They can be thought of as pointers to other prims. For
  example, a mesh prim can have a relationship to a material prim for shading.

A collection of these prims, with their attributes and relationships, is called a **USD stage**. It can be thought of
as a container for all prims in a scene. When we say we are designing a scene, we are actually designing a USD stage.

While working with direct USD APIs provides a lot of flexibility, it can be cumbersome to learn and use. To make it
easier to design scenes, Isaac Lab builds on top of the USD APIs to provide a configuration-driven interface to spawn prims
into a scene. These are included in the :mod:`sim.spawners` module.

Each prim is represented by an :class:`assets.AssetBaseCfg` that declares its path, spawn configuration, and initial
transform. :class:`TutorialCfg` owns all of these asset configurations alongside the simulation configuration. The
script gives that complete declarative input to a :class:`cloner.ReplicateSession`; the resulting clone plan supplies
the source paths used to spawn every asset.

.. literalinclude:: ../../../../scripts/tutorials/00_sim/spawn_prims.py
   :language: python
   :pyobject: TutorialCfg

For more information on the available spawn configurations, refer to the :mod:`sim.spawners` module.

.. attention::

   All the scene designing must happen before the simulation starts. Once the simulation starts, we recommend keeping
   the scene frozen and only altering the properties of the prim. This is particularly important for GPU simulation
   as adding new prims during simulation may alter the physics simulation buffers on GPU and lead to unexpected
   behaviors.


Spawning a ground plane
-----------------------

The :class:`~sim.spawners.from_files.GroundPlaneCfg` configures a grid-like ground plane with
modifiable properties such as its appearance and size. The tutorial assigns it to the ``ground`` asset configuration.


Spawning lights
---------------

It is possible to spawn `different light prims`_ into the stage. These include distant lights, sphere lights, disk
lights, and cylinder lights. In this tutorial, we spawn a distant light which is a light that is infinitely far away
from the scene and shines in a single direction. Its position is part of the ``light`` asset's initial-state
configuration.


Spawning primitive shapes
-------------------------

We configure cones using the :class:`~sim.spawners.shapes.ConeCfg` class. It is possible to specify
the radius, height, physics properties, and material properties of the cone. By default, the physics and material
properties are disabled. Their declared paths place them below ``/World/Objects``; the clone lifecycle creates the
required hierarchy.

The first two cones we spawn ``Cone1`` and ``Cone2`` are visual elements and do not have physics enabled.

For the third cone ``ConeRigid``, we add rigid body physics to it by setting the attributes for that in the configuration
class. Through these attributes, we can specify the mass, friction, and restitution of the cone. If unspecified, they
default to the default values set by USD Physics.

Lastly, we spawn a cuboid ``CuboidDeformable`` which contains deformable body physics properties. Unlike the
rigid body simulation, a deformable body can have relative motion between its vertices. This is useful for simulating
soft bodies like cloth, rubber, or jello. It is important to note that deformable bodies are only supported in
GPU simulation and require a mesh object to be spawned with deformable body physics properties and a deformable
physics material. This example uses the PhysX-specific deformable property and material cfgs.

Spawning from another file
--------------------------

Lastly, it is possible to spawn prims from other file formats such as other USD, URDF, or OBJ files. In this tutorial,
we spawn a USD file of a table into the scene. The table is a mesh prim and has a material prim associated with it.
All of this information is stored in its USD file.

The table above is added as a reference to the scene. In layman terms, this means that the table is not actually added
to the scene, but a ``pointer`` to the table asset is added. This allows us to modify the table asset and have the changes
reflected in the scene in a non-destructive manner. For example, we can change the material of the table without
actually modifying the underlying file for the table asset directly. Only the changes are stored in the USD stage.


Executing the Script
~~~~~~~~~~~~~~~~~~~~

Similar to the tutorial before, to run the script, execute the following command:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

        uv run --extra tetrahedralization python scripts/tutorials/00_sim/spawn_prims.py

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

        ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

Once the simulation starts, you should see a window with a ground plane, a light, some cones, and a table.
The green cone, which has rigid body physics enabled, should fall and collide with the table and the ground
plane. The other cones are visual elements and should not move. To stop the simulation, you can close the window,
or press ``Ctrl+C`` in the terminal.

.. figure:: ../../_static/tutorials/tutorial_spawn_prims.jpg
    :align: center
    :figwidth: 100%
    :alt: result of spawn_prims.py

This tutorial provided a foundation for spawning various prims into the scene in Isaac Lab. Although simple, it
demonstrates the basic concepts of scene designing in Isaac Lab and how to use the spawners. In the coming tutorials,
we will now look at how to interact with the scene and the simulation.


.. _`USD documentation`: https://openusd.org/release/index.html
.. _`different light prims`: https://youtu.be/c7qyI8pZvF4?feature=shared
