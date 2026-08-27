"""Explicit deterministic random streams for isolated behavioral runs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

_UINT64_MODULUS = 1 << 64
_UINT64_MASK = _UINT64_MODULUS - 1
_ROOT_SEED_MAX = _UINT64_MASK
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_STREAM_NAME_MAX_BYTES = 256


def _validate_root_seed(root_seed: object) -> int:
    if type(root_seed) is not int or not 0 <= root_seed <= _ROOT_SEED_MAX:
        raise ValueError("root_seed must be an unsigned 64-bit integer")
    return root_seed


def _validate_stream_name(stream_name: object) -> str:
    if type(stream_name) is not str or not stream_name:
        raise ValueError("stream_name must be a non-empty string")
    if len(stream_name.encode("utf-8")) > _STREAM_NAME_MAX_BYTES:
        raise ValueError(
            f"stream_name must not exceed {_STREAM_NAME_MAX_BYTES} UTF-8 bytes"
        )
    return stream_name


def derive_stream_seed(root_seed: int, stream_name: str) -> int:
    """Derive one independent 256-bit seed from a root seed and stream name."""
    root_seed = _validate_root_seed(root_seed)
    stream_name = _validate_stream_name(stream_name)
    material = f"{root_seed}:{stream_name}".encode()
    return int.from_bytes(sha256(material).digest(), byteorder="big")


@dataclass(frozen=True, slots=True)
class RngSnapshot:
    """Immutable continuation state for one named stream."""

    root_seed: int
    stream_name: str
    state: int

    def __post_init__(self) -> None:
        _validate_root_seed(self.root_seed)
        _validate_stream_name(self.stream_name)
        if type(self.state) is not int or not 0 <= self.state <= _UINT64_MASK:
            raise ValueError("state must be an unsigned 64-bit integer")


class NamedRng:
    """SplitMix64 generator owned by exactly one named stochastic component.

    Instances are deliberately mutable and must never be shared between isolated
    runs or worker threads. Use :class:`RngFactory` to construct a fresh stream.
    """

    __slots__ = ("_root_seed", "_seed", "_state", "_stream_name")

    def __init__(self, root_seed: int, stream_name: str) -> None:
        self._root_seed = _validate_root_seed(root_seed)
        self._stream_name = _validate_stream_name(stream_name)
        self._seed = derive_stream_seed(self._root_seed, self._stream_name)
        self._state = self._seed & _UINT64_MASK

    @classmethod
    def from_snapshot(cls, snapshot: RngSnapshot) -> NamedRng:
        """Restore an exact continuation without consuming a sample."""
        if type(snapshot) is not RngSnapshot:
            raise TypeError("snapshot must be an RngSnapshot")
        stream = cls(snapshot.root_seed, snapshot.stream_name)
        stream._state = snapshot.state
        return stream

    @property
    def root_seed(self) -> int:
        return self._root_seed

    @property
    def stream_name(self) -> str:
        return self._stream_name

    @property
    def seed(self) -> int:
        """Return the full SHA-256-derived seed as an integer."""
        return self._seed

    def snapshot(self) -> RngSnapshot:
        return RngSnapshot(
            root_seed=self._root_seed,
            stream_name=self._stream_name,
            state=self._state,
        )

    def next_u64(self) -> int:
        """Return the next unsigned 64-bit SplitMix64 sample."""
        self._state = (self._state + _SPLITMIX_GAMMA) & _UINT64_MASK
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
        return (value ^ (value >> 31)) & _UINT64_MASK

    def randbelow(self, stop: int) -> int:
        """Sample uniformly from ``range(stop)`` without modulo bias."""
        if type(stop) is not int or not 1 <= stop <= _UINT64_MODULUS:
            raise ValueError("stop must be an integer within [1, 2**64]")
        limit = _UINT64_MODULUS - (_UINT64_MODULUS % stop)
        while True:
            sample = self.next_u64()
            if sample < limit:
                return sample % stop

    def uniform(self) -> float:
        """Return a deterministic IEEE-754 value in the half-open interval [0, 1)."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))


@dataclass(frozen=True, slots=True)
class RngFactory:
    """Factory that prevents unrelated streams from sharing generator state."""

    root_seed: int

    def __post_init__(self) -> None:
        _validate_root_seed(self.root_seed)

    def stream(self, stream_name: str) -> NamedRng:
        return NamedRng(self.root_seed, stream_name)

    def world(self) -> NamedRng:
        return self.stream("world")

    def observations(self) -> NamedRng:
        return self.stream("observations")

    def candidate(self, module_name: str) -> NamedRng:
        module_name = _validate_stream_name(module_name)
        return self.stream(f"candidate:{module_name}")


__all__ = [
    "NamedRng",
    "RngFactory",
    "RngSnapshot",
    "derive_stream_seed",
]
