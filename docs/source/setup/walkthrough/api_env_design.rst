.. _walkthrough_api_env_design:

Classes and Configs
====================================

To begin, navigate to the task: ``source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial``, and take a look
and the contents of ``isaac_lab_tutorial_env_cfg.py``.  You should see something that looks like the following

.. code-block:: python

  from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

  import isaaclab.sim as sim_utils
  from isaaclab.assets import ArticulationCfg, AssetBaseCfg
  from isaaclab.envs import DirectRLEnvCfg
  from isaaclab.scene import InteractiveSceneCfg
  from isaaclab.sim import SimulationCfg
  from isaaclab.utils.configclass import configclass

  @configclass
  class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):

      # Some useful fields
      .
      .
      .

      # simulation
      sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)

      # assets
      robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
      ground_cfg: AssetBaseCfg = AssetBaseCfg(
          prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg()
      )
      light_cfg: AssetBaseCfg = AssetBaseCfg(
          prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
      )

      # scene
      scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0)

      # Some more useful fields
      .
      .
      .

This is the default configuration for a simple cartpole environment that comes with the template and defines the ``self`` scope
for anything you do within the corresponding environment.

.. currentmodule:: isaaclab.envs

The first thing to note is the presence of the ``@configclass`` decorator. This defines a class as a configuration class, which holds
a special place in Isaac Lab. Configuration classes are part of how Isaac Lab determines what to "care" about when it comes to cloning
the environment to scale up training. Isaac Lab provides different base configuration classes depending on your goals, and in this
case we are using the :class:`DirectRLEnvCfg` class because we are interested in performing reinforcement learning in the direct workflow.

.. currentmodule:: isaaclab.sim

The second thing to note is the content of the configuration class. The direct environment config is
the composition root: it contains the **sim**, asset configs such as the **robot**, and the **scene**
settings. These fields are also configuration classes. Keeping every authored asset config below this
root lets Isaac Lab describe the complete homogeneous scene before constructing its prototype.

The **sim** is an instance of :class:`SimulationCfg`, and this is the config that controls the nature of the simulated reality we are building. This field is a member
of the base class, ``DirecRLEnvCfg``, but has a default sim configuration, so it's *technically* optional.   The ``SimulationCfg`` dictates
how finely to step through time (dt), the direction of gravity, and even how physics should be simulated. In this case we only specify the time step and the render interval, with the
former indicating that each step through time should simulate :math:`1/120` th of a second, and the latter being how many steps we should take before we render a frame (a value of 2 means
render every other frame).

.. currentmodule:: isaaclab.scene

The **scene** is an instance of :class:`InteractiveSceneCfg`. In this homogeneous direct workflow it
holds the number of environments and their spacing; the asset configs remain on
``IsaacLabTutorialEnvCfg`` and are constructed in ``_setup_scene``. A scene subclass is useful instead
when a declarative or heterogeneous scene needs to own and construct its entities.

.. currentmodule:: isaaclab.assets

The environment's **robot** definition is an :class:`ArticulationCfg`. Here ``CARTPOLE_CFG`` comes
from ``isaaclab_assets.robots.cartpole``; replacing its path with ``{ENV_REGEX_NS}/Robot`` declares
one prototype path that the homogeneous clone lifecycle expands under every environment root.


The Environment
-----------------

Next, let's take a look at the contents of the other python file in our task directory: ``isaac_lab_tutorial_env.py``

.. code-block:: python

  # imports
  .
  .
  .
  from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg

  class IsaacLabTutorialEnv(DirectRLEnv):
      cfg: IsaacLabTutorialEnvCfg

      def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
          super().__init__(cfg, render_mode, **kwargs)
          . . .

      def _setup_scene(self):
          self.robot = self.cfg.robot_cfg.class_type(self.cfg.robot_cfg)
          self.cfg.ground_cfg.spawn.func(self.cfg.ground_cfg.prim_path, self.cfg.ground_cfg.spawn)
          self.cfg.light_cfg.spawn.func(self.cfg.light_cfg.prim_path, self.cfg.light_cfg.spawn)
          self.scene.articulations["robot"] = self.robot

      def _pre_physics_step(self, actions: torch.Tensor) -> None:
          . . .

      def _apply_action(self) -> None:
          . . .

      def _get_observations(self) -> dict:
          . . .

      def _get_rewards(self) -> torch.Tensor:
          total_reward = compute_rewards(...)
          return total_reward

      def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
          . . .

      def _reset_idx(self, env_ids: Sequence[int] | None):
          . . .

  @torch.jit.script
  def compute_rewards(...):
      . . .
      return total_reward


.. currentmodule:: isaaclab.envs

Some of the code has been omitted for clarity, in order to aid in discussion. This is where the actual "meat" of the
direct workflow exists and where most of our modifications will take place as we tweak the template to suit our needs.
Currently, all of the member functions of ``IsaacLabTutorialEnv`` are directly inherited from the :class:`DirectRLEnv`. This
known interface is how Isaac Lab and its supported RL frameworks interact with the environment.

When the environment is initialized, it passes its config to ``DirectRLEnv``. The base class builds
and publishes a cfg-derived homogeneous plan before constructing the plain scene and running
``_setup_scene``. The setup method constructs each cfg-owned prototype and registers runtime entities
with the scene; the base class then passes that same published plan to
:func:`~isaaclab.cloner.replicate`.

Notice also that the remaining functions do not take additional arguments except ``_reset_idx``.  This is because the environment only manages the application of
actions to the agent being simulated, and then updating the sim.  This is what the ``_pre_physics_step`` and ``_apply_action`` steps are for: we set the drive commands
to the robot so that when the simulation steps forward, the actions are applied and the joints are driven to new targets. This process is broken into steps like this
in order to ensure systematic control over how the environment is executed, and is especially important in the manager workflow. A similar relationship exists between the
``_get_dones`` function and ``_reset_idx``.  The former, ``_get_dones`` determines if each of the environments is in a terminal state, and populates tensors of boolean
values to indicate which environments terminated due to entering a terminal state vs time out (the two returned tensors of the function).  The latter, ``_reset_idx`` takes a
list environment index values (integers) and then actually resets those environments.  It is important that things like updating drive targets or resetting environments
do not happen **during** the physics or rendering steps, and breaking up the interface in this way helps prevent that.
