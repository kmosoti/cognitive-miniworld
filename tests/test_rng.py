"""MW-003 properties for explicit, independently named RNG streams."""

from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cmw.rng import NamedRng, RngFactory, derive_stream_seed

EXPECTED_SEED = (
    48_374_528_019_340_337_366_786_929_527_041_023_991_773_759_600_465_947_010_646_219_648_325_642_074_449
)
EXPECTED_SEQUENCE = (
    9_501_389_907_891_610_958,
    8_943_333_922_808_140_699,
    10_240_662_061_864_404_884,
    18_361_511_254_276_501_882,
)
STREAM_NAMES = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=16,
)


def _draw(stream: NamedRng, count: int = 32) -> tuple[int, ...]:
    return tuple(stream.next_u64() for _ in range(count))


def test_seed_and_algorithm_have_a_stable_snapshot() -> None:
    stream = NamedRng(7, "world")

    assert derive_stream_seed(7, "world") == EXPECTED_SEED
    assert _draw(stream, 4) == EXPECTED_SEQUENCE


def test_extra_draws_from_one_stream_do_not_shift_another() -> None:
    factory = RngFactory(41)
    expected_world = _draw(factory.world())

    observation_stream = factory.observations()
    _draw(observation_stream, 10_000)

    assert _draw(factory.world()) == expected_world
    assert _draw(factory.candidate("planner")) != expected_world


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(
    root_seed=st.integers(min_value=0, max_value=(1 << 64) - 1),
    names=st.lists(STREAM_NAMES, min_size=2, max_size=8, unique=True),
)
def test_reordered_concurrent_scheduling_preserves_each_stream(
    root_seed: int,
    names: list[str],
) -> None:
    factory = RngFactory(root_seed)
    expected = {name: _draw(factory.stream(name)) for name in names}

    def scheduled(name: str) -> tuple[str, tuple[int, ...]]:
        return name, _draw(factory.stream(name))

    with ThreadPoolExecutor(max_workers=min(4, len(names))) as executor:
        actual = dict(executor.map(scheduled, reversed(names)))

    assert actual == expected


def test_snapshot_restores_the_exact_continuation() -> None:
    stream = NamedRng(23, "candidate:planner")
    _draw(stream, 13)
    snapshot = stream.snapshot()

    expected = _draw(stream)
    restored = NamedRng.from_snapshot(snapshot)

    assert _draw(restored) == expected


@pytest.mark.parametrize("stop", [1, 2, 3, 10, (1 << 64) - 1, 1 << 64])
def test_randbelow_stays_inside_its_requested_range(stop: int) -> None:
    stream = NamedRng(5, f"range-{stop}")

    assert all(0 <= stream.randbelow(stop) < stop for _ in range(200))


@pytest.mark.parametrize(
    ("root_seed", "stream_name"),
    [
        (-1, "world"),
        (1 << 64, "world"),
        (True, "world"),
        (1, ""),
        (1, 7),
    ],
)
def test_invalid_stream_identity_is_rejected(
    root_seed: object,
    stream_name: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        NamedRng(cast(int, root_seed), cast(str, stream_name))


@pytest.mark.parametrize("stop", [0, -1, True, (1 << 64) + 1])
def test_invalid_randbelow_bound_is_rejected(stop: object) -> None:
    with pytest.raises(ValueError):
        NamedRng(1, "world").randbelow(cast(int, stop))
