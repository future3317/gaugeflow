import torch

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
