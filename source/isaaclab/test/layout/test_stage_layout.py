# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CSR semantics tests for :class:`isaaclab.layout.StageLayout`.

Pure tensor logic; no AppLauncher / USD dependency.
"""

from __future__ import annotations

import pytest
import torch

from isaaclab.layout import StageLayout, first_world_of, make_stage_layout, world_ids_of


def _identity_pose(num_envs: int, device: str = "cpu") -> torch.Tensor:
    """Return ``[num_envs, 7]`` poses at the origin with identity quaternion (xyzw)."""
    pose = torch.zeros((num_envs, 7), dtype=torch.float32, device=device)
    pose[:, 6] = 1.0
    return pose


class _Source:
    """Identity-keyed marker; the layout never inspects source internals."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid only
        return f"_Source({self.name!r})"


def test_stage_layout_dataclass_fields():
    """StageLayout exposes exactly the documented six fields."""
    layout = StageLayout(
        sources=(),
        destinations=(),
        source_ids=torch.empty(0, dtype=torch.long),
        destination_ids=torch.empty(0, dtype=torch.long),
        world_start=torch.tensor([0, 0], dtype=torch.long),
        env_pose=_identity_pose(0),
    )
    expected = {"sources", "destinations", "source_ids", "destination_ids", "world_start", "env_pose"}
    assert set(layout.__dataclass_fields__) == expected


def test_make_stage_layout_homogeneous_two_envs():
    """One source in every env produces CSR offsets ``[0, 0, 1, 2]``."""
    src = _Source("a")
    layout = make_stage_layout(
        sources=[src],
        destinations=["/World/envs/env_{}/A"],
        sources_per_world=[[], [0], [0]],
        env_pose=_identity_pose(2),
    )

    assert layout.world_start.tolist() == [0, 0, 1, 2]
    assert layout.source_ids.tolist() == [0, 0]
    assert layout.destination_ids.tolist() == [0, 0]
    # CSR boundary invariants
    assert int(layout.world_start[0]) == 0
    assert int(layout.world_start[-1]) == int(layout.source_ids.numel())


def test_make_stage_layout_shared_scope_only():
    """A source living only in shared scope occupies world 0 with no env slots."""
    src = _Source("ground")
    layout = make_stage_layout(
        sources=[src],
        destinations=["/World/Ground"],
        sources_per_world=[[0], [], []],
        env_pose=_identity_pose(2),
    )

    assert layout.world_start.tolist() == [0, 1, 1, 1]
    assert layout.source_ids.tolist() == [0]


def test_make_stage_layout_heterogeneous_multi_per_env():
    """Two sources in one env, plus a shared source, exercise full CSR."""
    shared = _Source("ground")
    a = _Source("a")
    b = _Source("b")
    layout = make_stage_layout(
        sources=[shared, a, b],
        destinations=["/World/Ground", "/World/envs/env_{}/A", "/World/envs/env_{}/B"],
        sources_per_world=[[0], [1], [1, 2]],
        env_pose=_identity_pose(2),
    )

    assert layout.source_ids.tolist() == [0, 1, 1, 2]
    assert layout.destination_ids.tolist() == [0, 1, 1, 2]
    assert layout.world_start.tolist() == [0, 1, 2, 4]


def test_make_stage_layout_independent_destination_ids():
    """``destinations_per_world`` may diverge from ``sources_per_world`` (shared scope + per-env)."""
    shared = _Source("ground")
    layout = make_stage_layout(
        sources=[shared],
        destinations=["/World/Ground", "/World/envs/env_{}/Ground"],
        sources_per_world=[[0], [0], [0]],
        destinations_per_world=[[0], [1], [1]],
        env_pose=_identity_pose(2),
    )

    assert layout.source_ids.tolist() == [0, 0, 0]
    assert layout.destination_ids.tolist() == [0, 1, 1]


def test_make_stage_layout_validation_errors():
    """Mismatched lengths / shapes / out-of-range indices all raise ``ValueError``."""
    src = _Source("a")

    with pytest.raises(ValueError, match=r"env_pose"):
        make_stage_layout([src], ["/A"], [[0]], torch.zeros(2, dtype=torch.float32))

    with pytest.raises(ValueError, match=r"sources_per_world must have length"):
        make_stage_layout([src], ["/A"], [[0]], _identity_pose(2))

    with pytest.raises(ValueError, match=r"destinations_per_world must be parallel"):
        make_stage_layout(
            [src],
            ["/A"],
            [[0], [0], []],
            _identity_pose(2),
            destinations_per_world=[[0], [0]],
        )

    with pytest.raises(ValueError, match=r"slot-for-slot"):
        make_stage_layout(
            [src],
            ["/A"],
            [[0], [0], []],
            _identity_pose(2),
            destinations_per_world=[[0, 0], [0], []],
        )

    with pytest.raises(ValueError, match=r"out-of-range source"):
        make_stage_layout([src], ["/A"], [[], [5], []], _identity_pose(2))

    with pytest.raises(ValueError, match=r"out-of-range destination"):
        make_stage_layout([src], ["/A"], [[], [0], []], _identity_pose(2), destinations_per_world=[[], [9], []])


def test_first_world_of_shared_per_env_and_missing():
    """``first_world_of`` distinguishes shared scope (-1), per-env (>=0), and missing (None)."""
    shared = _Source("shared")
    a = _Source("a")
    b = _Source("b")
    missing = _Source("missing")
    layout = make_stage_layout(
        sources=[shared, a, b],
        destinations=["/World/G", "/World/envs/env_{}/A", "/World/envs/env_{}/B"],
        sources_per_world=[[0], [], [1], [2, 1]],
        env_pose=_identity_pose(3),
    )

    assert first_world_of(layout, shared) == -1
    assert first_world_of(layout, a) == 1
    assert first_world_of(layout, b) == 2
    assert first_world_of(layout, missing) is None


def test_first_world_of_unregistered_returns_none():
    """A source present in :attr:`StageLayout.sources` but with no slots returns ``None``."""
    a = _Source("a")
    b = _Source("b")
    layout = make_stage_layout(
        sources=[a, b],
        destinations=["/World/A", "/World/B"],
        sources_per_world=[[], [0], [0]],
        env_pose=_identity_pose(2),
    )
    assert first_world_of(layout, b) is None


def test_world_ids_of_includes_duplicates_for_cardinality_gt_one():
    """Two slots in one env yield two equal entries; the env_id repeats."""
    a = _Source("a")
    b = _Source("b")
    layout = make_stage_layout(
        sources=[a, b],
        destinations=["/World/envs/env_{}/A", "/World/envs/env_{}/B"],
        sources_per_world=[[], [0, 0], [0, 1]],
        env_pose=_identity_pose(2),
    )

    assert world_ids_of(layout, a).tolist() == [0, 0, 1]
    assert world_ids_of(layout, b).tolist() == [1]


def test_world_ids_of_shared_scope_returns_minus_one():
    """Slots in world 0 (shared) decode to env_id ``-1``."""
    a = _Source("a")
    layout = make_stage_layout(
        sources=[a],
        destinations=["/World/Shared"],
        sources_per_world=[[0, 0], [], []],
        env_pose=_identity_pose(2),
    )
    assert world_ids_of(layout, a).tolist() == [-1, -1]


def test_world_ids_of_missing_or_empty_returns_empty_tensor():
    """Unknown source and registered-but-unslotted source both return an empty tensor."""
    a = _Source("a")
    b = _Source("b")
    missing = _Source("missing")
    layout = make_stage_layout(
        sources=[a, b],
        destinations=["/A", "/B"],
        sources_per_world=[[], [0], [0]],
        env_pose=_identity_pose(2),
    )
    assert world_ids_of(layout, missing).numel() == 0
    assert world_ids_of(layout, b).numel() == 0


def test_layout_is_identity_keyed_not_value_keyed():
    """Two distinct sources that compare equal still resolve to their own slots."""

    class _EqSource:
        def __init__(self, n):
            self.n = n

        def __eq__(self, other):
            return isinstance(other, _EqSource) and self.n == other.n

        def __hash__(self):
            return hash(self.n)

    src_a = _EqSource(0)
    src_b = _EqSource(0)  # equal-by-value but distinct identity
    layout = make_stage_layout(
        sources=[src_a, src_b],
        destinations=["/A", "/B"],
        sources_per_world=[[], [0], [1]],
        env_pose=_identity_pose(2),
    )

    assert first_world_of(layout, src_a) == 0
    assert first_world_of(layout, src_b) == 1
    assert world_ids_of(layout, src_a).tolist() == [0]
    assert world_ids_of(layout, src_b).tolist() == [1]


def test_csr_invariants_hold_under_random_construction():
    """``world_start`` is monotone non-decreasing, starts at 0, ends at len(source_ids)."""
    rng = torch.Generator().manual_seed(0)
    sources = [_Source(f"s{i}") for i in range(4)]
    destinations = [f"/W/{i}" for i in range(4)]
    num_envs = 6

    sources_per_world: list[list[int]] = []
    for _ in range(num_envs + 1):
        n = int(torch.randint(0, 4, (1,), generator=rng).item())
        sources_per_world.append(torch.randint(0, 4, (n,), generator=rng).tolist())

    layout = make_stage_layout(
        sources=sources,
        destinations=destinations,
        sources_per_world=sources_per_world,
        env_pose=_identity_pose(num_envs),
    )

    assert layout.world_start.shape == (num_envs + 2,)
    assert int(layout.world_start[0]) == 0
    assert int(layout.world_start[-1]) == int(layout.source_ids.numel())
    diffs = layout.world_start[1:] - layout.world_start[:-1]
    assert torch.all(diffs >= 0)

    expected_lengths = torch.tensor([len(slot) for slot in sources_per_world], dtype=torch.long)
    assert torch.equal(diffs, expected_lengths)
