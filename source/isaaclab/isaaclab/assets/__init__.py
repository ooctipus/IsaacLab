# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for scene assets, such as lights, rigid objects, and articulations.

An :class:`Asset` authors one plan-owned scene asset. :class:`AssetBase` extends it with the
physics handles and data buffers needed to interact with runtime simulation state.

Construction uses the active clone plan to author the prototype paths assigned to the asset. The
clone lifecycle then populates their declared destinations. See :attr:`AssetBaseCfg.spawn` and
:mod:`isaaclab.sim.spawners` for authoring configuration.

Runtime asset classes register callbacks that construct physics handles when simulation starts and
provide optional debug visualization through :attr:`AssetBaseCfg.debug_vis`.

The asset class follows the following naming convention for its methods:

* **set_xxx()**: These are used to only set the buffers into the :attr:`data` instance. However, they
  do not write the data into the simulator. The writing of data only happens when the
  :meth:`write_data_to_sim` method is called.
* **write_xxx_to_sim()**: These are used to set the buffers into the :attr:`data` instance and write
  the corresponding data into the simulator as well.
* **update(dt)**: These are used to update the buffers in the :attr:`data` instance. This should
  be called after a simulation step is performed.

The main reason to separate the ``set`` and ``write`` operations is to provide flexibility to the
user when they need to perform a post-processing operation of the buffers before applying them
into the simulator. A common example for this is dealing with explicit actuator models where the
specified joint targets are not directly applied to the simulator but are instead used to compute
the corresponding actuator torques.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
