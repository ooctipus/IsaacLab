# octibenchmark Conventions

## NVTX Label Naming

Labels follow the pattern `{component}.{method}` or `{component}.{feature}:{term_name}`:

```
sim.step                                    # simulation step
sim.render                                  # render call
reward.compute                              # reward manager compute
reward.term:reach_distance                  # individual reward term
observation.term[proprioceptive]:joint_pos  # observation term with group
vision.compute_image_obs                    # task-specific via extra_nvtx_hooks
```

## Tag Format

Run tags are used as filenames and wandb labels:

```
{task}__envs{N}__{axis1}_{value1}__...__{launcher}[__phase]
```

Example:
```
Shadow-Vision__envs4096__newton_newton_renderer_rgb__128_128__non_rl
```

Rules:
- Task name stripped of common prefixes (`Isaac-`, `Repose-Cube-`, `-v0`)
- Environment count always included
- Each sweep axis (sorted alphabetically) with compacted values
- Launcher type (`non_rl` or `rsl_rl`)
- Phase only if non-default (i.e., `startup`)

### Value compaction
- `width=128 height=128` → `128_128`
- `presets=newton,newton_renderer,rgb` → `newton_newton_renderer_rgb`

## Hydra Overrides

- **hydra_sweeps**: Dict of `{axis_name: [override_string, ...]}`. Each axis value
  can contain multiple space-separated Hydra overrides if they belong together.
- **hydra_args**: List of constant overrides applied to every run (not swept).

## Example Matrix Definition

```python
from octibenchmark.bench_cfg import BenchmarkMatrix, Launcher

matrix = BenchmarkMatrix(
    tasks=["Isaac-Repose-Cube-Shadow-Vision-Direct-v0"],
    num_envs=[2048, 4096, 8192],
    hydra_sweeps={
        "preset": [
            "presets=newton,newton_renderer,rgb",
            "presets=newton,newton_renderer,depth",
        ],
        "resolution": [
            "env.tiled_camera.width=64 env.tiled_camera.height=64",
            "env.tiled_camera.width=128 env.tiled_camera.height=128",
        ],
    },
    hydra_args=["env.decimation=4"],
    launcher=Launcher.NON_RL,
    num_frames=100,
    warmup_frames=10,
    extra_nvtx_hooks=[
        ("_compute_image_observations", "vision.image_obs"),
        ("feature_extractor.step", "vision.feature_extractor"),
    ],
)

# Generates: 1 task × 3 num_envs × 2 presets × 2 resolutions = 12 runs
ALL_MATRICES = {"my_vision_bench": matrix}
```

## Kernel Pattern Conventions

Patterns are Python regexes applied via `re.search` (substring, case-insensitive):

| Question | Pattern |
|---|---|
| Physics solver breakdown | `"ccd"`, `"broadphase"`, `"linesearch"`, `"cholesky"` |
| Indexed vs masked memory | `"index\|scatter\|index_put"`, `"mask\|where"` |
| Rendering cost | `"ray_trace\|render"`, `"rasterize"` |

## Preset Combinations

Not all preset combinations are valid for all tasks. Known gotchas:

| Task | Working preset | Failing preset | Why |
|---|---|---|---|
| `Isaac-Dexsuite-Kuka-Allegro-Lift-v0` | `presets=newton,cube` | `presets=cube` | Needs a physics backend (newton) |

When defining matrices with multiple presets, expect some combinations to fail.
Always check `run_matrix.py` output for errors and proceed with successful runs.

## Testing Conventions

- Uses standard `unittest` framework
- Tests validate data transformation and command building — no actual benchmark execution
- Test files: `test_nvtx_hooks.py`, `test_analyze.py`, `test_bench_cfg.py`
- Run via: `./isaaclab.sh -p -m pytest scripts/octibenchmark/test_*.py`
