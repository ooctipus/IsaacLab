# IsaacLab Development Guidelines
- Make the smallest change that solves the requested problem.
- Preserve existing behavior unless the task explicitly changes it.
- Read surrounding code, callers, tests, and documentation before changing an interface.
- Do not add dependencies unless necessary. Prefer existing project dependencies and the standard library.
- Do not commit generated plans, scratch files, or agent artifacts.
- Follow a more-specific `AGENTS.md` in the directory being changed.
- Use the repository's current SPDX header template for new source files; do not change existing file headers.
- Follow the existing style and abstractions in the affected package.
- Use modern Python type hints, including `X | None` instead of `Optional[X]`.
- Use `snake_case` for methods, functions, and CLI arguments.
- Keep related public symbols discoverable through consistent prefixes.
- Keep source-prim uniqueness and match-count validation in the shared resolver; assets and sensors
  must not repair duplicate query results independently.
- Keep Newton solver schema registration in the active manager's builder factory; the cloner must not depend on solver modules.
- Resolve Newton raycast BVH requirements before builder finalization; sensor task registration must not add a late BVH fallback.
- Keep joint-wrench sensor coverage separate from articulation control-joint selection. Reuse cached
  body bindings without changing the shared view's joint filters or creating a second view for sensing.
- Keep articulation ordering maps on articulation data; do not mirror maps or add cached ordering flags.
- Keep backend ownership on `SimulationContext`, using backend type and configuration rather than service
  locators, resource keys, or separate renderer registries.
- Renderers consume geometry through `SceneDataProvider`. Keep Newton imports out of OVRTX renderers
  and Fabric destination ownership and shadow remapping out of physics backends.
- For external wrenches, follow the asset API's `is_global` boolean and `_b`/`_w` buffer naming. Keep
  frame conversion decisions in `WrenchComposer` and track pending contributions with plain booleans;
  do not introduce frame enums, content bitmasks, or a classification layer.
- Name wrench reads `get_forces_and_torques`, matching the existing add/set methods; avoid a separate
  "submission" API or compatibility alias for the unreleased `resolve_submission` method.
- Use concrete types for public interfaces where practical.
- For task-local Newton selection, resolve names in the path utility before numeric binding. Bound selections must not retain path expressions or accept `int | str` identities. Preserve ordered occurrence relations, including repeated asset instances.
- Compose the public Newton handle/placement relations with task-owned environment participation. Do not reimplement lifetime validation, inspect private population storage, or infer the same MuJoCo coordinate conversion in separate consumers.
- State every selection count's axis explicitly: world counts and prototype counts are different relations. Validate joint correspondence using owner, domain and ordered joint identity, not equal counts alone.
- Validate selected fields against the prepared model's authoritative attribute frequency. Matching integer indices or scalar shapes do not make coordinate, DOF and body fields interchangeable; do not duplicate the model's schema in a task registry.
- Validate explicit indexed writes before changing state: environment indices must be unique, in range and on the owner device, and values must match the declared shape and dtype. Keep the full-population path free of index scans.
- Fused MDP kernels must preserve tensor shape and index-domain contracts before dereferencing native storage; broadcasting is not a pointer-bound guarantee.
- Mixed Torch/Warp task roots own producer ordering across actions, physics, resets and observations. Use one shared stream-scope operation with dependencies on both entry and exit; do not rely on default-stream coincidence or scatter task synchronization into numerical kernels.
- Capture owners choose a private capturable stream and order it on entry and exit; eager operations and replay honor the caller stream, including Torch's legacy default stream.
- Private numerical MDP helpers preserve the task root's stream ordering; they must not impose whole-device synchronization.
- The environment owns terminal-observation timing; task overrides own successor previews and must preserve live command state, RNG and observation history. Do not copy a parent step loop to customize terminal observations.
- Keep the original integer width when comparing identities. Masked identities may be arbitrary; included out-of-range identities must be rejected without accessing the destination bank.
- Borrow derived body poses through read-only field descriptors with the existing placement, readiness and episode masks. Do not cache gathered physics values or expose derived poses as independent writable coordinates.
- Keep environment position, world ID plus generation, prototype index and storage slot distinct. Name conversion maps by destination and source index domains.
- Name reset operations by their effect: requesting, staging and publishing are distinct. Do not give a staging-only operation and an immediate population rebuild the same contract.
- Equal registered variants do not make mutable physical properties constant: reset must restore them after changes. Elide host resource publication only under sole-writer ownership and exact object-identity proof against every variant and initial destination.
- A retained Python owner does not make explicitly retired native storage usable. Reject selection access after owner retirement or runtime closure before allocating or launching work; raw borrowed descriptors remain scoped to that lifetime.
- Admit selection metadata, runtime storage and world handles only on one device before allocating descriptor tables. A nested pointer descriptor does not establish cross-device accessibility.
- Retain the exact buffers consumed by captured task kernels. Do not retain preparation callbacks or the whole task root as a substitute for explicit resource ownership.
- Use Google-style docstrings for public APIs.
- Document SI units for public physical quantities in docstrings using inline `[unit]` notation (e.g. `Particle positions [m], shape [N, 3]`); use `[m or rad, depending on joint type]` where applicable, and skip non-physical fields (indices, counts, flags).
- Keep comments brief and explain intent, constraints, or edge cases rather than restating code.
- Do not remove or rename a public API without a prior deprecation and migration path.
- Update public documentation when adding or changing public APIs.
- Verify documented technical claims against the current code and primary sources before relying on them.
- Use the uv-managed environment for routine commands.
- Use `uv run python` for Python scripts and tests.
- Use `uv run isaaclab` for Isaac Lab CLI commands.
- Use `./isaaclab.sh` only for installer workflows that require it.
- Do not define Warp kernels in `python -c`; write a temporary Python file instead so Warp can inspect the source.

## Code style

- Group imports in PEP 8 order, separated by blank lines: `__future__`, standard library, third-party,
  Omniverse runtime packages (`isaacsim`, `omni`, `pxr`, `carb`, ...), Isaac Lab packages, then local relative imports.
  Ruff enforces this order through `uv run isaaclab -f`; do not sort imports by hand.
- Import from the same package with relative imports when the target is at most three leading dots away
  (e.g. `from ...utils import math as math_utils`). Use absolute imports for deeper targets and for other packages.
- Keep absolute imports in modules that can run as scripts (with an `if __name__ == "__main__":` block),
  since relative imports fail there.

## Testing and validation

- Run the narrowest relevant test first.
- Before reporting that a test cannot run because an optional dependency is missing, identify its project extra and retry with `uv run --extra <extra> ...`.
- Do not treat an optional dependency missing from the base environment as a blocker when its project extra is available.
- Run `uv run isaaclab -f` before committing.
- For a regression test, verify that it fails without the fix and passes with it.
- Find and extend the closest existing test before creating a new test file or test case.
- Add a test only when it covers a distinct behavior, regression, boundary, or failure mode that existing tests do not cover clearly.
- Test observable behavior and public contracts, not implementation details.
- Validate joint-wrench frames with the same physical fixture and analytic load expectations across backends.
  Do not derive the expected wrench by repeating the production transformation on the backend's raw output.
- Use hard-coded values only when they are the intended contract or a small, independently verified example; otherwise derive the expected result from a separate, simple reference calculation.
- Keep tests focused and remove or consolidate redundant coverage instead of growing overlapping test suites.
- Do not add debug output to production Warp kernels. Use temporary standalone reproductions and remove debug output before committing.

## Changelog and release metadata

- Do not edit `CHANGELOG.rst` or `config/extension.toml` directly.
- Add one changelog fragment for each changed source package when the change is user-visible.
- Use `.skip` fragments for changes that do not require a release note.
- Write changelog entries in past tense and include migration guidance for deprecated, changed, or removed behavior.
- Mark breaking changes clearly and provide migration guidance.

## Commits and branches

- Work on a feature branch; do not commit directly to `main`.
- Keep commits focused and atomic.
- Use an imperative, capitalized commit subject with no trailing period.
- Inspect staged changes before committing.
- Do not add AI co-author or attribution lines.
- Prefer follow-up commits over amending commits while addressing review feedback.

## Repository skills

- Keep repository-owned skills in `skills/`; do not duplicate their contents in tool-specific discovery directories.
- Validate skill changes with `uv run --no-project python tools/skills/cli.py check`.
- Keep skills concise and point to maintained documentation and source examples.

## Schema fragment consumers

- When migrating a spawner schema slot to fragments, migrate its readers and overrides too.
  Use a bare fragment for a single instance and a list only for multiple fragments. For lists,
  select the owning fragment before accessing fields or calling `replace()`; `fix_root_link`
  belongs on the spawner.
- For file-spawned fixtures that must only tune existing physics bodies, use explicit fragment
  target mappings. A bare fragment or list may create a missing body and change the fixture's validity.
