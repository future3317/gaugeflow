import torch

from gaugeflow.production.balanced_rank_sharded_data import BalancedRankShardedStream
from gaugeflow.production.rank_sharded_data import ExactRankShardedStream


def _pair(*, wrap: bool = False) -> tuple[ExactRankShardedStream, ExactRankShardedStream]:
    common = dict(dataset_size=11, global_batch_size=4, world_size=2, seed=17, wrap=wrap)
    return (
        ExactRankShardedStream(rank=0, **common),
        ExactRankShardedStream(rank=1, **common),
    )


def test_rank_sharded_stream_covers_one_permutation_without_padding() -> None:
    first, second = _pair()
    reconstructed: list[int] = []
    local_counts = [0, 0]
    while not first.exhausted:
        left = first.next_indices()
        right = second.next_indices()
        local_counts[0] += left.numel()
        local_counts[1] += right.numel()
        interleaved = torch.empty(left.numel() + right.numel(), dtype=torch.long)
        interleaved[0::2] = left
        interleaved[1::2] = right
        reconstructed.extend(interleaved.tolist())
    assert sorted(reconstructed) == list(range(11))
    assert local_counts == [6, 5]
    assert first.state.global_examples_emitted == 11
    assert second.state.global_examples_emitted == 11
    assert second.exhausted


def test_rank_sharded_stream_resume_and_wrapped_replay_are_exact() -> None:
    first, _ = _pair(wrap=True)
    first.next_indices()
    first.next_indices()
    state = first.state_dict()
    reference = [first.next_indices(), first.next_indices(), first.next_indices()]
    resumed, _ = _pair(wrap=True)
    resumed.load_state_dict(state)
    repeated = [resumed.next_indices(), resumed.next_indices(), resumed.next_indices()]
    assert all(torch.equal(left, right) for left, right in zip(reference, repeated, strict=True))
    assert resumed.state == first.state


def _balanced_pair() -> tuple[BalancedRankShardedStream, BalancedRankShardedStream]:
    source = torch.tensor([0] * 7 + [1] * 5 + [2] * 3, dtype=torch.uint8)
    common = dict(
        source_index=source,
        source_weights=(0.2, 0.3, 0.5),
        global_batch_size=6,
        world_size=2,
        seed=29,
    )
    return (
        BalancedRankShardedStream(rank=0, **common),
        BalancedRankShardedStream(rank=1, **common),
    )


def test_balanced_stream_reconstructs_global_batches_and_source_mixture() -> None:
    first, second = _balanced_pair()
    source = torch.tensor([0] * 7 + [1] * 5 + [2] * 3)
    observed: list[torch.Tensor] = []
    for _ in range(200):
        left = first.next_indices()
        right = second.next_indices()
        global_indices = torch.empty(6, dtype=torch.long)
        global_indices[0::2] = left
        global_indices[1::2] = right
        observed.append(source[global_indices])
    fraction = torch.bincount(torch.cat(observed), minlength=3).float() / 1200
    assert torch.allclose(fraction, torch.tensor([0.2, 0.3, 0.5]), atol=0.03)
    assert first.state.global_examples_emitted == 1200
    assert first.state.local_examples_emitted == 600
    assert first.state.source_epochs == second.state.source_epochs
    assert first.state.source_offsets == second.state.source_offsets


def test_balanced_stream_resume_is_exact_and_binds_source_partition() -> None:
    first, _ = _balanced_pair()
    for _ in range(5):
        first.next_indices()
    state = first.state_dict()
    reference = [first.next_indices() for _ in range(8)]
    resumed, _ = _balanced_pair()
    resumed.load_state_dict(state)
    repeated = [resumed.next_indices() for _ in range(8)]
    assert all(torch.equal(left, right) for left, right in zip(reference, repeated, strict=True))
    assert resumed.state == first.state

    changed = BalancedRankShardedStream(
        torch.tensor([0] * 6 + [1] * 6 + [2] * 3, dtype=torch.uint8),
        (0.2, 0.3, 0.5),
        6,
        rank=0,
        world_size=2,
        seed=29,
    )
    try:
        changed.load_state_dict(state)
    except ValueError as error:
        assert "configuration mismatch" in str(error)
    else:
        raise AssertionError("balanced stream accepted a different source partition")


def test_balanced_stream_block_locality_and_checkpoint_binding() -> None:
    source = torch.zeros(16, dtype=torch.uint8)
    blocks = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
    common = dict(
        source_index=source,
        source_weights=(1.0,),
        global_batch_size=4,
        rank=0,
        world_size=1,
        seed=41,
        block_index=blocks,
    )
    stream = BalancedRankShardedStream(**common)
    batches = [stream.next_indices() for _ in range(2)]
    assert all(torch.unique(blocks[batch]).numel() == 1 for batch in batches)
    observed = torch.cat(batches)
    assert torch.unique(observed).numel() == observed.numel()

    state = stream.state_dict()
    reference = stream.next_indices()
    resumed = BalancedRankShardedStream(**common)
    resumed.load_state_dict(state)
    assert torch.equal(resumed.next_indices(), reference)

    changed = BalancedRankShardedStream(**{**common, "block_index": blocks.roll(1)})
    try:
        changed.load_state_dict(state)
    except ValueError as error:
        assert "configuration mismatch" in str(error)
    else:
        raise AssertionError("balanced stream accepted a different block partition")
