# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Property tests for :class:`CSRGraph`.

These tests pin two things:

1. **Adapter invariants** -- the producer/consumer naming layer behaves
   as documented (dedup is order-preserving, ``-1`` marks inactive
   consumers, representatives are the first-encountered consumer per
   producer).
2. **Underlying BSR contract** -- ``BsrMatrix`` exposes the CSR
   structure we expect (``offsets`` monotonic, ``offsets[-1] == nnz``,
   ``columns`` enumerates active consumers per row).

Test 9 (``test_byte_identity_vs_inline_legacy``) is the regression
guardrail: it replays the legacy ``_build_signature_tables`` algorithm
inline on the same input and asserts identical Warp output. If
``CSRGraph`` ever silently diverges, that test fails.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from isaaclab_tasks.core.multi_task.mdp.commands.impl.primitive_graph_local.csr_graph import CSRGraph

# Use CPU device throughout -- the build path is CPU-only Python; no need to
# require a GPU for these tests.
_DEVICE = "cpu"


def _to_list(arr: wp.array) -> list[int]:
    """Pull a Warp int32 array down to a Python list for assertions."""
    return arr.numpy().tolist()


@pytest.fixture(autouse=True, scope="module")
def _warp_init():
    wp.init()


# ---------------------------------------------------------------------------
# Adapter invariants.
# ---------------------------------------------------------------------------


def test_empty_keys() -> None:
    """All-``None`` keys yield zero producers and an all-``-1`` consumer map."""
    g = CSRGraph.build_from_consumer_keys([None, None, None], device=_DEVICE)
    assert g.num_consumer_slots == 3
    assert g.num_active_consumers == 0
    assert g.num_producers == 0
    assert _to_list(g.consumer_to_producer) == [-1, -1, -1]
    assert _to_list(g.producer_to_representative_consumer) == []


def test_zero_slot_graph() -> None:
    """An empty consumer list yields a zero-sized graph."""
    g = CSRGraph.build_from_consumer_keys([], device=_DEVICE)
    assert g.num_consumer_slots == 0
    assert g.num_active_consumers == 0
    assert g.num_producers == 0


def test_single_active_consumer() -> None:
    """One active consumer creates exactly one producer, mapping to itself."""
    g = CSRGraph.build_from_consumer_keys([(7, 3)], device=_DEVICE)
    assert g.num_consumer_slots == 1
    assert g.num_active_consumers == 1
    assert g.num_producers == 1
    assert _to_list(g.consumer_to_producer) == [0]
    assert _to_list(g.producer_to_representative_consumer) == [0]


def test_dedup_groups_identical_keys() -> None:
    """Identical keys collapse to one producer; representative is the first."""
    keys = [(1, 2), (1, 2), (1, 2)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    assert g.num_producers == 1
    assert g.num_active_consumers == 3
    assert _to_list(g.consumer_to_producer) == [0, 0, 0]
    assert _to_list(g.producer_to_representative_consumer) == [0]


def test_distinct_keys_each_get_own_producer() -> None:
    """N distinct keys yield N producers in first-encounter order."""
    keys = [(1,), (2,), (3,), (4,)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    assert g.num_producers == 4
    assert _to_list(g.consumer_to_producer) == [0, 1, 2, 3]
    assert _to_list(g.producer_to_representative_consumer) == [0, 1, 2, 3]


def test_inactive_interspersed_with_active() -> None:
    """Inactive consumers don't disturb producer-id assignment or representatives."""
    keys = [None, (5, 5), None, (6, 6), (5, 5), None]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    assert g.num_consumer_slots == 6
    assert g.num_active_consumers == 3
    assert g.num_producers == 2
    assert _to_list(g.consumer_to_producer) == [-1, 0, -1, 1, 0, -1]
    assert _to_list(g.producer_to_representative_consumer) == [1, 3]


def test_insertion_order_determinism() -> None:
    """Producer ids reflect the order keys are *first* seen, not their hash."""
    # (9, 9) appears before (1, 1) even though it'd hash-sort after.
    keys = [(9, 9), (1, 1), (9, 9), (1, 1)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    assert _to_list(g.consumer_to_producer) == [0, 1, 0, 1]
    assert _to_list(g.producer_to_representative_consumer) == [0, 1]


# ---------------------------------------------------------------------------
# Underlying BsrMatrix / CSR invariants.
# ---------------------------------------------------------------------------


def test_csr_offsets_are_monotonic_nondecreasing() -> None:
    """CSR offsets must be non-decreasing and bracket ``nnz``."""
    keys = [(1,), (2,), (1,), (3,), (2,), (1,)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    offsets = np.asarray(g.producer_offsets.numpy())
    assert offsets.shape == (g.num_producers + 1,)
    assert np.all(np.diff(offsets) >= 0), f"offsets not monotonic: {offsets.tolist()}"
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == g.num_active_consumers


def test_csr_offsets_last_equals_nnz() -> None:
    """``offsets[num_producers] == nnz == num_active_consumers`` always."""
    g = CSRGraph.build_from_consumer_keys([(0,), (1,), (1,), (2,), (0,), None, (2,)], device=_DEVICE)
    offsets = g.producer_offsets.numpy()
    assert int(offsets[-1]) == g.bsr.nnz == g.num_active_consumers


def test_csr_columns_partition_active_consumers() -> None:
    """``producer_consumers`` is a permutation of the active consumer ids."""
    keys = [(0,), None, (1,), (0,), None, (2,), (1,)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    active_cids = sorted([cid for cid, k in enumerate(keys) if k is not None])
    columns_sorted = sorted(g.producer_consumers.numpy().tolist())
    assert columns_sorted == active_cids


def test_csr_columns_match_consumer_to_producer() -> None:
    """For each (producer, consumer) edge in the CSR, ``consumer_to_producer``
    must agree -- this is the round-trip property between dense and CSR views.
    """
    keys = [(0,), (1,), (1,), (2,), (0,), None, (2,), (2,)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    offsets = g.producer_offsets.numpy()
    columns = g.producer_consumers.numpy()
    consumer_to_producer = g.consumer_to_producer.numpy()
    for p in range(g.num_producers):
        for i in range(int(offsets[p]), int(offsets[p + 1])):
            c = int(columns[i])
            assert int(consumer_to_producer[c]) == p, (
                f"CSR edge (producer={p}, consumer={c}) disagrees with"
                f" consumer_to_producer[{c}]={int(consumer_to_producer[c])}"
            )


def test_representative_consumer_is_in_producers_consumer_set() -> None:
    """Each producer's representative must be one of its own consumers."""
    keys = [(0,), (1,), (0,), (1,), (0,), (2,)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    offsets = g.producer_offsets.numpy()
    columns = g.producer_consumers.numpy()
    reps = g.producer_to_representative_consumer.numpy()
    for p in range(g.num_producers):
        consumer_set = set(int(c) for c in columns[int(offsets[p]) : int(offsets[p + 1])])
        assert int(reps[p]) in consumer_set


# ---------------------------------------------------------------------------
# Graph-theoretic introspection (Phase 2).
# ---------------------------------------------------------------------------


def test_introspection_empty_graph() -> None:
    """All introspection methods handle the empty graph without sync."""
    g = CSRGraph.build_from_consumer_keys([None, None, None], device=_DEVICE)
    assert g.max_fanout == 0
    assert g.mean_fanout == 0.0
    assert g.fanout_histogram == {}


def test_max_fanout_picks_largest_row() -> None:
    """``max_fanout`` is the largest per-producer consumer count."""
    keys = [(0,), (0,), (0,), (1,), (2,), (2,)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    assert g.max_fanout == 3


def test_mean_fanout_is_edge_count_over_producers() -> None:
    """``mean_fanout`` = num_active_consumers / num_producers."""
    keys = [(0,), (0,), (1,), (1,), (1,), (2,)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    assert g.num_active_consumers == 6
    assert g.num_producers == 3
    assert g.mean_fanout == pytest.approx(2.0)


def test_fanout_histogram_for_uniform_workload() -> None:
    """Uniform fan-out gives a single-entry histogram."""
    # Three producers, two consumers each.
    keys = [(0,), (0,), (1,), (1,), (2,), (2,)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    assert g.fanout_histogram == {2: 3}


def test_fanout_histogram_for_skewed_workload() -> None:
    """Skewed fan-out gives multi-entry histogram with weighted distribution."""
    # Producer 0 has 4 consumers, producer 1 has 1, producer 2 has 2.
    keys = [(0,), (0,), (0,), (0,), (1,), (2,), (2,)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    assert g.fanout_histogram == {1: 1, 2: 1, 4: 1}


def test_max_fanout_consistent_with_histogram_max_bin() -> None:
    """``max_fanout`` equals the largest key in :attr:`fanout_histogram`."""
    keys = [(0,), (0,), (1,), (1,), (1,), (1,), (2,)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    assert g.max_fanout == max(g.fanout_histogram.keys())


def test_introspection_with_inactive_consumers() -> None:
    """Inactive consumers don't pollute fan-out stats — they're not edges."""
    keys = [(0,), None, (0,), None, (1,)]
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    assert g.num_active_consumers == 3
    assert g.num_producers == 2
    assert g.max_fanout == 2  # producer 0
    assert g.mean_fanout == pytest.approx(1.5)
    assert g.fanout_histogram == {1: 1, 2: 1}


# ---------------------------------------------------------------------------
# Byte-identity vs the legacy inline algorithm (regression guardrail).
# ---------------------------------------------------------------------------


def _inline_legacy_build(
    consumer_keys: list[tuple[int, ...] | None],
) -> tuple[list[int], list[int]]:
    """Reproduces ``_build_signature_tables`` semantics on raw inputs.

    Returns ``(consumer_to_producer, representative_consumer_per_producer)``.
    This is the exact algorithm primitive_graph_local has shipped with;
    keeping a duplicate here protects against an accidental semantic shift
    in :meth:`CSRGraph.build_from_consumer_keys`.
    """
    producer_id_by_key: dict[tuple[int, ...], int] = {}
    representative: list[int] = []
    out_consumer_to_producer: list[int] = [-1] * len(consumer_keys)
    for cid, key in enumerate(consumer_keys):
        if key is None:
            continue
        pid = producer_id_by_key.get(key)
        if pid is None:
            pid = len(representative)
            producer_id_by_key[key] = pid
            representative.append(cid)
        out_consumer_to_producer[cid] = pid
    return out_consumer_to_producer, representative


@pytest.mark.parametrize(
    "keys",
    [
        [(0,), (1,), (0,), (2,), (1,), None, (2,)],
        [(1, 2), (1, 2), (3, 4), None, (1, 2), (3, 4), (5, 6)],
        [None, None, (7,), (7,), None, (7,)],
        [(i % 3, i % 5) for i in range(50)],
        [],
        [(0,)],
    ],
)
def test_byte_identity_vs_inline_legacy(keys: list[tuple[int, ...] | None]) -> None:
    """``CSRGraph`` output must match the inline legacy algorithm exactly."""
    expected_c2p, expected_rep = _inline_legacy_build(keys)
    g = CSRGraph.build_from_consumer_keys(keys, device=_DEVICE)
    assert _to_list(g.consumer_to_producer) == expected_c2p
    assert _to_list(g.producer_to_representative_consumer) == expected_rep
