"""Deterministic source-balanced rank shards for continued pretraining."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class BalancedRankShardedStreamState:
    batch: int
    source_epochs: tuple[int, ...]
    source_offsets: tuple[int, ...]
    global_examples_emitted: int
    local_examples_emitted: int


class BalancedRankShardedStream:
    """Draw a source mixture and stride each deterministic global batch by rank.

    Every source is traversed without replacement until its own permutation is
    exhausted. Sources wrap independently, so balancing changes exposure but
    never pads a distributed batch or duplicates an example inside a source
    epoch. All ranks reproduce the same global batch and retain a stride.
    """

    def __init__(
        self,
        source_index: torch.Tensor,
        source_weights: Sequence[float],
        global_batch_size: int,
        *,
        rank: int,
        world_size: int,
        seed: int,
        block_index: torch.Tensor | None = None,
    ) -> None:
        if (
            source_index.ndim != 1
            or source_index.numel() < 1
            or source_index.dtype not in {torch.uint8, torch.int16, torch.int32, torch.int64}
            or global_batch_size < 1
            or world_size < 1
            or rank < 0
            or rank >= world_size
            or seed < 0
        ):
            raise ValueError("balanced rank-sharded stream configuration is invalid")
        source_count = len(source_weights)
        weights = torch.as_tensor(source_weights, dtype=torch.float64)
        if (
            source_count < 1
            or weights.shape != (source_count,)
            or not bool(torch.isfinite(weights).all())
            or bool((weights <= 0.0).any())
            or int(source_index.min()) < 0
            or int(source_index.max()) >= source_count
        ):
            raise ValueError("balanced stream source vocabulary or weights are invalid")
        self.source_index = source_index.cpu().contiguous().long()
        self.source_weights = (weights / weights.sum()).tolist()
        self.global_batch_size = global_batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        if block_index is not None and (
            block_index.ndim != 1
            or block_index.shape != source_index.shape
            or block_index.dtype not in {torch.int16, torch.int32, torch.int64}
            or int(block_index.min()) < 0
        ):
            raise ValueError("balanced stream block partition is invalid")
        self.block_index = (
            block_index.cpu().contiguous().long() if block_index is not None else None
        )
        self._source_members = tuple(
            torch.nonzero(self.source_index == source, as_tuple=False).squeeze(1)
            for source in range(source_count)
        )
        if any(members.numel() == 0 for members in self._source_members):
            raise ValueError("balanced stream cannot sample an empty source")
        if self.block_index is not None:
            block_count = int(self.block_index.max()) + 1
            block_sources = torch.full((block_count,), -1, dtype=torch.long)
            for source, members in enumerate(self._source_members):
                blocks = torch.unique(self.block_index[members])
                if bool((block_sources[blocks] >= 0).any()):
                    raise ValueError("balanced stream block crosses source boundaries")
                block_sources[blocks] = source
            if bool((block_sources < 0).any()):
                raise ValueError("balanced stream block vocabulary is not compact")
            grouped_blocks: list[tuple[torch.Tensor, ...]] = []
            for members in self._source_members:
                member_blocks = self.block_index[members]
                order = torch.argsort(member_blocks, stable=True)
                sorted_members = members[order]
                _, counts = torch.unique_consecutive(
                    member_blocks[order], return_counts=True
                )
                grouped_blocks.append(tuple(torch.split(sorted_members, counts.tolist())))
            self._source_blocks: tuple[tuple[torch.Tensor, ...], ...] | None = tuple(
                grouped_blocks
            )
        else:
            self._source_blocks = None
        self._source_digest = hashlib.sha256(self.source_index.numpy().tobytes()).hexdigest()
        self._block_digest = (
            hashlib.sha256(self.block_index.numpy().tobytes()).hexdigest()
            if self.block_index is not None
            else None
        )
        self._batch = 0
        self._source_epochs = [0] * source_count
        self._source_offsets = [0] * source_count
        self._global_examples_emitted = 0
        self._local_examples_emitted = 0
        self._source_permutations = [
            self._make_source_permutation(source) for source in range(source_count)
        ]

    def _make_source_permutation(self, source: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * self._source_epochs[source] + 10_007 * source
        )
        members = self._source_members[source]
        if self._source_blocks is None:
            return members[torch.randperm(members.numel(), generator=generator)]
        blocks = self._source_blocks[source]
        block_order = torch.randperm(len(blocks), generator=generator).tolist()
        return torch.cat(
            [
                blocks[index][torch.randperm(blocks[index].numel(), generator=generator)]
                for index in block_order
            ]
        )

    def _take_source(self, source: int, count: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        while count > 0:
            members = self._source_members[source]
            offset = self._source_offsets[source]
            take = min(count, members.numel() - offset)
            chunks.append(self._source_permutations[source][offset : offset + take])
            self._source_offsets[source] += take
            count -= take
            if self._source_offsets[source] == members.numel():
                self._source_epochs[source] += 1
                self._source_offsets[source] = 0
                self._source_permutations[source] = self._make_source_permutation(source)
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)

    @property
    def state(self) -> BalancedRankShardedStreamState:
        return BalancedRankShardedStreamState(
            batch=self._batch,
            source_epochs=tuple(self._source_epochs),
            source_offsets=tuple(self._source_offsets),
            global_examples_emitted=self._global_examples_emitted,
            local_examples_emitted=self._local_examples_emitted,
        )

    def next_indices(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + 97_003 * self._batch + 53)
        choices = torch.multinomial(
            torch.tensor(self.source_weights, dtype=torch.float64),
            self.global_batch_size,
            replacement=True,
            generator=generator,
        )
        global_indices = torch.empty(self.global_batch_size, dtype=torch.long)
        counts = torch.bincount(choices, minlength=len(self.source_weights))
        for source, count in enumerate(counts.tolist()):
            if count:
                global_indices[choices == source] = self._take_source(source, count)
        local_indices = global_indices[self.rank :: self.world_size].contiguous()
        self._batch += 1
        self._global_examples_emitted += global_indices.numel()
        self._local_examples_emitted += local_indices.numel()
        return local_indices

    def state_dict(self) -> dict[str, Any]:
        state = self.state
        return {
            "schema": 2,
            "source_digest": self._source_digest,
            "block_digest": self._block_digest,
            "source_weights": self.source_weights,
            "global_batch_size": self.global_batch_size,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "batch": state.batch,
            "source_epochs": list(state.source_epochs),
            "source_offsets": list(state.source_offsets),
            "global_examples_emitted": state.global_examples_emitted,
            "local_examples_emitted": state.local_examples_emitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schema": 2,
            "source_digest": self._source_digest,
            "block_digest": self._block_digest,
            "source_weights": self.source_weights,
            "global_batch_size": self.global_batch_size,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
        }
        if any(state.get(name) != value for name, value in expected.items()):
            raise ValueError("balanced stream checkpoint configuration mismatch")
        batch = state.get("batch")
        epochs = state.get("source_epochs")
        offsets = state.get("source_offsets")
        global_emitted = state.get("global_examples_emitted")
        local_emitted = state.get("local_examples_emitted")
        integers = (batch, global_emitted, local_emitted)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise ValueError("balanced stream checkpoint counters are invalid")
        if not isinstance(epochs, list) or not isinstance(offsets, list) or not (
            len(epochs) == len(offsets) == len(self._source_members)
        ):
            raise ValueError("balanced stream checkpoint source cursors are invalid")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in epochs + offsets):
            raise ValueError("balanced stream checkpoint source cursors are invalid")
        assert isinstance(batch, int) and isinstance(global_emitted, int)
        assert isinstance(local_emitted, int)
        if batch < 0 or global_emitted != batch * self.global_batch_size:
            raise ValueError("balanced stream global counter disagrees with batch cursor")
        local_per_batch = len(range(self.rank, self.global_batch_size, self.world_size))
        if local_emitted != batch * local_per_batch:
            raise ValueError("balanced stream local counter disagrees with batch cursor")
        for source, (epoch, offset, members) in enumerate(
            zip(epochs, offsets, self._source_members, strict=True)
        ):
            if epoch < 0 or not 0 <= offset < members.numel():
                raise ValueError(f"balanced stream source {source} cursor is invalid")
        consumed = sum(
            epoch * members.numel() + offset
            for epoch, offset, members in zip(epochs, offsets, self._source_members, strict=True)
        )
        if consumed != global_emitted:
            raise ValueError("balanced stream source cursors disagree with global counter")
        self._batch = batch
        self._source_epochs = list(epochs)
        self._source_offsets = list(offsets)
        self._global_examples_emitted = global_emitted
        self._local_examples_emitted = local_emitted
        self._source_permutations = [
            self._make_source_permutation(source) for source in range(len(self._source_members))
        ]
