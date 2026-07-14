from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from time import monotonic
from typing import Generic, Optional, TypeVar


K = TypeVar("K")
V = TypeVar("V")


def cache_key(value: str) -> str:
    """Return a stable, process-independent key for in-memory caches."""
    return sha256(value.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hit: int = 0
    miss: int = 0
    evicted: int = 0
    expired: int = 0


class TTLCache(Generic[K, V]):
    """Thread-safe in-memory LRU + TTL cache."""

    def __init__(self, max_size: int = 4096, ttl_seconds: int = 1800):
        self.max_size = max(1, int(max_size))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._lock = Lock()
        self.stats = CacheStats()

    def _is_expired(self, expires_at: float) -> bool:
        return monotonic() > expires_at

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.stats.miss += 1
                return None

            expires_at, value = item
            if self._is_expired(expires_at):
                self._data.pop(key, None)
                self.stats.miss += 1
                self.stats.expired += 1
                return None

            self._data.move_to_end(key, last=True)
            self.stats.hit += 1
            return value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            expires_at = monotonic() + self.ttl_seconds
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key, last=True)

            while len(self._data) > self.max_size:
                self._data.popitem(last=False)
                self.stats.evicted += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
