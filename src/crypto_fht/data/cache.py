"""
Data caching utilities.

Provides file-based and in-memory caching for API responses
and computed values to reduce redundant fetches.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached value with metadata.

    Attributes:
        value: The cached value.
        created_at: When the entry was created.
        expires_at: When the entry expires (None for no expiration).
        key: Cache key.
    """

    value: T
    created_at: datetime
    expires_at: datetime | None
    key: str

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class DataCache:
    """File-based data cache with expiration.

    Stores cached data as pickle files in a specified directory.

    Attributes:
        cache_dir: Directory for cache files.
        default_ttl: Default time-to-live for entries.

    Example:
        >>> cache = DataCache(cache_dir=Path("~/.crypto_fht/cache"))
        >>> cache.set("reserves", reserves_data, ttl=timedelta(hours=1))
        >>> data = cache.get("reserves")
    """

    def __init__(
        self,
        cache_dir: Path | str = "~/.crypto_fht/cache",
        default_ttl: timedelta = timedelta(hours=1),
    ) -> None:
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files.
            default_ttl: Default TTL for entries.
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for frequently accessed items
        self._memory_cache: dict[str, CacheEntry[Any]] = {}

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        # Hash the key for safe filenames
        key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
        safe_key = "".join(c if c.isalnum() else "_" for c in key[:32])
        return self.cache_dir / f"{safe_key}_{key_hash}.pkl"

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get value from cache.

        Args:
            key: Cache key.
            default: Default value if not found or expired.

        Returns:
            Cached value or default.
        """
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired:
                return entry.value
            else:
                del self._memory_cache[key]

        # Check file cache
        path = self._key_to_path(key)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    entry: CacheEntry[T] = pickle.load(f)

                if not entry.is_expired:
                    # Promote to memory cache
                    self._memory_cache[key] = entry
                    return entry.value
                else:
                    # Clean up expired entry
                    path.unlink()
            except (pickle.PickleError, EOFError, OSError):
                # Corrupted cache file
                path.unlink(missing_ok=True)

        return default

    def set(
        self,
        key: str,
        value: T,
        ttl: timedelta | None = None,
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live. Uses default_ttl if None.
        """
        if ttl is None:
            ttl = self.default_ttl

        now = datetime.now()
        expires_at = now + ttl if ttl.total_seconds() > 0 else None

        entry = CacheEntry(
            value=value,
            created_at=now,
            expires_at=expires_at,
            key=key,
        )

        # Store in memory
        self._memory_cache[key] = entry

        # Store on disk
        path = self._key_to_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(entry, f)
        except (OSError, pickle.PickleError):
            # Caching failure is not critical
            pass

    def delete(self, key: str) -> bool:
        """Delete entry from cache.

        Args:
            key: Cache key.

        Returns:
            True if entry was deleted.
        """
        deleted = False

        if key in self._memory_cache:
            del self._memory_cache[key]
            deleted = True

        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            deleted = True

        return deleted

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        count = len(self._memory_cache)
        self._memory_cache.clear()

        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()
            count += 1

        return count

    def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed.
        """
        count = 0

        # Memory cache
        expired_keys = [
            k for k, v in self._memory_cache.items() if v.is_expired
        ]
        for k in expired_keys:
            del self._memory_cache[k]
            count += 1

        # File cache
        for path in self.cache_dir.glob("*.pkl"):
            try:
                with open(path, "rb") as f:
                    entry = pickle.load(f)
                if entry.is_expired:
                    path.unlink()
                    count += 1
            except (pickle.PickleError, EOFError, OSError):
                path.unlink(missing_ok=True)
                count += 1

        return count

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        file_count = len(list(self.cache_dir.glob("*.pkl")))
        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.pkl")
        )

        return {
            "memory_entries": len(self._memory_cache),
            "file_entries": file_count,
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
        }


class InMemoryCache(Generic[T]):
    """Simple in-memory cache with LRU eviction.

    Suitable for caching computed values during a session.

    Attributes:
        max_size: Maximum number of entries.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._cache: dict[str, tuple[T, datetime]] = {}
        self._access_order: list[str] = []

    def get(self, key: str) -> T | None:
        """Get value from cache."""
        if key in self._cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key][0]
        return None

    def set(self, key: str, value: T) -> None:
        """Set value in cache."""
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]

        self._cache[key] = (value, datetime.now())
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)
