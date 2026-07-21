"""Exact no-padding rank shards for deterministic production data streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RankShardedStreamState:
    epoch: int
    offset: int
    global_examples_emitted: int
    local_examples_emitted: int


class ExactRankShardedStream:
    """Reproduce one global permutation on every rank and take strided shards.

    The stream never pads a tail.  Wrapped streams concatenate the end of one
    permutation with the start of the next, which is appropriate for replay;
    non-wrapped streams expose every dataset row exactly once.
    """

    def __init__(
        self,
        dataset_size: int,
        global_batch_size: int,
        *,
        rank: int,
        world_size: int,
        seed: int,
        wrap: bool,
    ) -> None:
        if (
            dataset_size < 1
            or global_batch_size < 1
            or world_size < 1
            or rank < 0
            or rank >= world_size
            or seed < 0
        ):
            raise ValueError("rank-sharded stream configuration is invalid")
        self.dataset_size = dataset_size
        self.global_batch_size = global_batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.wrap = wrap
        self._epoch = 0
        self._offset = 0
        self._global_examples_emitted = 0
        self._local_examples_emitted = 0
        self._permutation = self._make_permutation()

    def _make_permutation(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * self._epoch)
        return torch.randperm(self.dataset_size, generator=generator)

    @property
    def exhausted(self) -> bool:
        return not self.wrap and self._offset == self.dataset_size

    @property
    def state(self) -> RankShardedStreamState:
        return RankShardedStreamState(
            epoch=self._epoch,
            offset=self._offset,
            global_examples_emitted=self._global_examples_emitted,
            local_examples_emitted=self._local_examples_emitted,
        )

    def next_indices(self) -> torch.Tensor:
        if self.exhausted:
            raise StopIteration
        chunks: list[torch.Tensor] = []
        requested = self.global_batch_size
        while requested > 0:
            available = self.dataset_size - self._offset
            take = min(requested, available)
            chunks.append(self._permutation[self._offset : self._offset + take])
            self._offset += take
            requested -= take
            if self._offset == self.dataset_size:
                if not self.wrap:
                    break
                self._epoch += 1
                self._offset = 0
                self._permutation = self._make_permutation()
        global_indices = torch.cat(chunks)
        local_indices = global_indices[self.rank :: self.world_size].contiguous()
        self._global_examples_emitted += global_indices.numel()
        self._local_examples_emitted += local_indices.numel()
        return local_indices

    def state_dict(self) -> dict[str, Any]:
        state = self.state
        return {
            "schema": 1,
            "dataset_size": self.dataset_size,
            "global_batch_size": self.global_batch_size,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "wrap": self.wrap,
            "epoch": state.epoch,
            "offset": state.offset,
            "global_examples_emitted": state.global_examples_emitted,
            "local_examples_emitted": state.local_examples_emitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schema": 1,
            "dataset_size": self.dataset_size,
            "global_batch_size": self.global_batch_size,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "wrap": self.wrap,
        }
        if any(state.get(name) != value for name, value in expected.items()):
            raise ValueError("rank-sharded stream checkpoint configuration mismatch")
        values = tuple(
            state.get(name)
            for name in (
                "epoch",
                "offset",
                "global_examples_emitted",
                "local_examples_emitted",
            )
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("rank-sharded stream checkpoint counters are invalid")
        raw_epoch, raw_offset, raw_global, raw_local = values
        assert isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool)
        assert isinstance(raw_offset, int) and not isinstance(raw_offset, bool)
        assert isinstance(raw_global, int) and not isinstance(raw_global, bool)
        assert isinstance(raw_local, int) and not isinstance(raw_local, bool)
        epoch, offset, global_emitted, local_emitted = (
            raw_epoch,
            raw_offset,
            raw_global,
            raw_local,
        )
        if (
            epoch < 0
            or not 0 <= offset <= self.dataset_size
            or global_emitted < 0
            or local_emitted < 0
            or (self.wrap and offset == self.dataset_size)
            or (not self.wrap and epoch != 0)
        ):
            raise ValueError("rank-sharded stream checkpoint cursor is invalid")
        expected_global = epoch * self.dataset_size + offset
        full_batches, tail = divmod(global_emitted, self.global_batch_size)
        local_per_batch = len(range(self.rank, self.global_batch_size, self.world_size))
        expected_local = full_batches * local_per_batch + len(
            range(self.rank, tail, self.world_size)
        )
        if global_emitted != expected_global or local_emitted != expected_local:
            raise ValueError("rank-sharded stream checkpoint counters disagree with cursor")
        self._epoch = epoch
        self._offset = offset
        self._global_examples_emitted = global_emitted
        self._local_examples_emitted = local_emitted
        self._permutation = self._make_permutation()
