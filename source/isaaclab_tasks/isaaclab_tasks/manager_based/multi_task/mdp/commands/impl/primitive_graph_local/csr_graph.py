# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CSR-graph adapter for primitive_graph_local's producer-consumer fan-out.

Wraps :class:`warp.sparse.BsrMatrix` (CSR storage at 1x1 blocks) under a
producer/consumer naming layer. The adapter exists so application code can
talk about producers and consumers while storage stays Warp-native.

Why an adapter and not a fresh CSR primitive: Warp ships a battle-tested
COO->CSR build in :mod:`warp.sparse` (``BsrMatrix`` + ``bsr_from_triplets``).
Reusing it gives us:

* the C++ native CSR build for free, with the dedup / row-grouping
  invariants documented by the sparse-matrix literature.
* automatic inheritance of any future Warp-side optimizations
  (cuGraph-style tensor-core-aware formats are on NVIDIA's roadmap).
* a maintenance surface of ~50 LOC instead of a parallel CSR implementation.

The adapter exists because ``BsrMatrix`` uses linear-algebra vocabulary
(rows/cols/nnz/offsets). Our domain layers two things on top:

1. **Producer/consumer naming.** ``BsrMatrix`` rows are producers (unique
   signatures), columns are consumers (subtask slots), values are unused
   (we pass 1 to satisfy the BSR contract).
2. **Sparse subtask-id indexing.** Each consumer either points to one
   producer or is *inactive* (its state kernel isn't in this schedule
   group). The adapter exposes a ``consumer_to_producer`` array indexed
   by the global subtask id, with ``-1`` for inactive consumers --
   matching the legacy ``_build_signature_tables`` output exactly so
   primitive_graph_local kernels keep their existing signatures.

Reference: NVIDIA Warp's BSR implementation lives in
``warp._src/sparse.py:90`` (``BsrMatrix``), ``:355`` (``bsr_zeros``), and
``:481-655`` (``bsr_from_triplets`` / ``bsr_set_from_triplets``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import warp as wp
import warp.sparse as wps

__all__ = ["CSRGraph"]


@dataclass
class CSRGraph:
    """Many-to-one producer-consumer fan-out in CSR storage.

    Each *active* consumer reads from exactly one producer; *inactive*
    consumers (those whose state kernel falls outside this schedule
    group) carry ``-1`` in :attr:`consumer_to_producer` and contribute
    no CSR edge.

    Attributes:
        consumer_to_producer: ``int32`` array of length
            :attr:`num_consumer_slots`. Entry ``c`` is the producer id
            consumer ``c`` reads from, or ``-1`` if consumer ``c`` is
            inactive in this graph.
        producer_to_representative_consumer: ``int32`` array of length
            :attr:`num_producers`. Entry ``p`` is one consumer slot that
            depends on producer ``p`` -- the first one encountered during
            build, used by kernels that need a single source-index per
            producer (e.g. to read the shared state slice once).
        bsr: Underlying :class:`warp.sparse.BsrMatrix` (1x1 blocks,
            ``int32`` values). Rows are producers, columns are consumers,
            ``nnz`` equals the number of active consumers. Held for
            graph-theoretic introspection and producer-coherent
            reordering in later phases.
    """

    consumer_to_producer: wp.array
    producer_to_representative_consumer: wp.array
    bsr: wps.BsrMatrix

    @property
    def num_consumer_slots(self) -> int:
        """Total consumer slot count, including inactive slots."""
        return int(self.consumer_to_producer.shape[0])

    @property
    def num_active_consumers(self) -> int:
        """Number of consumers that actually edge into a producer (``nnz``)."""
        return int(self.bsr.nnz)

    @property
    def num_producers(self) -> int:
        """Number of unique producers."""
        return int(self.producer_to_representative_consumer.shape[0])

    @property
    def producer_offsets(self) -> wp.array:
        """CSR row offsets, length ``num_producers + 1``.

        Producer ``p`` owns consumers in the half-open slice
        ``[offsets[p], offsets[p+1])`` of :attr:`producer_consumers`.
        """
        return self.bsr.offsets

    @property
    def producer_consumers(self) -> wp.array:
        """CSR column-index array (flat consumer ids), length
        :attr:`num_active_consumers`.
        """
        return self.bsr.columns

    @classmethod
    def build_from_consumer_keys(
        cls,
        consumer_keys: Sequence[tuple[int, ...] | None],
        device: str | wp.context.Device,
    ) -> CSRGraph:
        """Build a :class:`CSRGraph` by hash-deduping consumer keys.

        Producer ids are assigned in *first-encounter* order (Python dict
        insertion order), matching the legacy ``_build_signature_tables``
        semantics so callers that switch to this adapter see byte-identical
        producer-id assignments.

        Args:
            consumer_keys: One entry per consumer slot. Each entry is
                either a tuple of ints (the consumer's gather signature)
                or ``None`` to mark the consumer inactive.
            device: Warp device string or :class:`~warp.context.Device`
                for the resulting arrays.

        Returns:
            A new :class:`CSRGraph` whose adjacency reflects the deduped
            signatures.
        """
        device_str = str(device)
        num_consumer_slots = len(consumer_keys)

        producer_id_by_key: dict[tuple[int, ...], int] = {}
        representative_consumer: list[int] = []
        consumer_to_producer: list[int] = [-1] * num_consumer_slots
        for cid, key in enumerate(consumer_keys):
            if key is None:
                continue
            producer_id = producer_id_by_key.get(key)
            if producer_id is None:
                producer_id = len(representative_consumer)
                producer_id_by_key[key] = producer_id
                representative_consumer.append(cid)
            consumer_to_producer[cid] = producer_id

        num_producers = len(representative_consumer)
        active_cids = [cid for cid, pid in enumerate(consumer_to_producer) if pid != -1]
        active_pids = [consumer_to_producer[cid] for cid in active_cids]
        num_edges = len(active_cids)

        consumer_to_producer_wp = wp.array(consumer_to_producer, dtype=wp.int32, device=device_str)
        producer_to_representative_consumer_wp = wp.array(representative_consumer, dtype=wp.int32, device=device_str)

        # An empty graph (no active edges) skips bsr_from_triplets -- that path
        # requires non-empty value arrays. bsr_zeros gives the right shape.
        if num_edges == 0:
            bsr = wps.bsr_zeros(
                rows_of_blocks=num_producers,
                cols_of_blocks=num_consumer_slots,
                block_type=wp.int32,
                device=device_str,
            )
        else:
            rows_wp = wp.array(active_pids, dtype=wp.int32, device=device_str)
            columns_wp = wp.array(active_cids, dtype=wp.int32, device=device_str)
            # Values are unused by the adjacency; pass 1s so prune_numerical_zeros
            # never drops a valid edge.
            values_wp = wp.array([1] * num_edges, dtype=wp.int32, device=device_str)
            bsr = wps.bsr_from_triplets(
                rows_of_blocks=num_producers,
                cols_of_blocks=num_consumer_slots,
                rows=rows_wp,
                columns=columns_wp,
                values=values_wp,
                prune_numerical_zeros=False,
            )

        return cls(
            consumer_to_producer=consumer_to_producer_wp,
            producer_to_representative_consumer=producer_to_representative_consumer_wp,
            bsr=bsr,
        )
