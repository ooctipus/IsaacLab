# IsaacLab Development Guidelines

## Contribution guide

Read the relevant sections of [the contribution guide](docs/source/refs/contributing.rst) before starting:

- [Coding Style](docs/source/refs/contributing.rst#coding-style) for implementation, refactoring, and review.
- [Unit Testing](docs/source/refs/contributing.rst#unit-testing) for test changes and validation;
  use the [test-audit skill](skills/developer/test-audit/SKILL.md) when adding, changing, reviewing, or pruning tests.
- [Contributing Documentation](docs/source/refs/contributing.rst#contributing-documentation) for documentation changes.
- [Maintaining package changelogs and versions](docs/source/refs/contributing.rst#maintaining-package-changelogs-and-versions)
  for source package changes.
- [Tools](docs/source/refs/contributing.rst#tools) for formatting and lint checks.

The guide owns shared contribution rules. Update them there instead of copying them into this file or skills.

## Agent workflow

- Follow a more-specific `AGENTS.md` in the directory being changed.
- Preserve unrelated workspace changes and do not commit generated plans, scratch files, or agent artifacts.
- Use the repository's current SPDX header template for new source files; do not change existing file headers.
- Follow the existing style and abstractions in the affected package.
- Use the uv-managed environment for routine commands and `uv run python` for Python scripts.
- Run the guide's formatting and lint checks before committing.
- Do not define Warp kernels in `python -c`; write a temporary Python file instead so Warp can inspect the source.
- Do not add debug output to production Warp kernels. Use temporary standalone reproductions and remove debug output before committing.

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

## Dynamic keyboard integration

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
