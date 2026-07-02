# Isaac Lab Newton Simulation interfaces

## Newton-native MJCF assets

Use `NewtonMjcfFileCfg` when a Newton-backed scene should parse a local MJCF
file directly, without first converting it to USD through Kit:

```python
from isaaclab.assets import ArticulationCfg
from isaaclab_newton.sim import NewtonMjcfFileCfg

robot = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=NewtonMjcfFileCfg(asset_path="/absolute/path/to/robot.xml"),
    articulation_root_prim_path="/model_name",
)
```

The spawner authors a lightweight USD marker. During Newton model
construction, the active solver parses the MJCF and owns its equality-constraint
policy. Core `isaaclab.sim.MjcfFileCfg` remains the Kit-based MJCF-to-USD
conversion boundary.
