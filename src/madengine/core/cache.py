#!/usr/bin/env python3
"""Module for run caching functionality.

This module provides caching capabilities for madengine runs, including:
- Docker image tar archiving
- Run configuration bundling
- LRU cache management with size limits
- Multi-node support

Classes:
    CacheEntry: Represents a single cache entry
    CacheManager: Manages cache entries and eviction

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import os
import json
import hashlib
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import fcntl

# third-party modules
from madengine.core.console import Console


@dataclass
class CacheEntry:
    """Represents a cached run entry.

    Attributes:
        cache_id: Unique identifier for this cache entry
        cache_key: Hash-based key for cache matching
        docker_image: Docker image name
        docker_tar_path: Path to saved docker tar file
        run_config: Model run configuration
        model_tags: Tags used for this run
        created_at: Creation timestamp (ISO format)
        last_used: Last access timestamp (ISO format)
        size_bytes: Total size on disk
        node_count: Number of nodes (1 for single-node, >1 for multi-node)
        cache_dir: Directory containing this cache entry
    """
    cache_id: str
    cache_key: str
    docker_image: str
    docker_tar_path: str
    run_config: Dict[str, Any]
    model_tags: List[str]
    created_at: str
    last_used: str
    size_bytes: int
    node_count: int
    cache_dir: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create cache entry from dictionary."""
        return cls(**data)

    def update_last_used(self) -> None:
        """Update the last used timestamp to now."""
        self.last_used = datetime.now().isoformat()


class CacheManager:
    """Manages cache entries for madengine runs.

    Provides functionality for:
    - Saving and loading cache entries
    - LRU eviction policy
    - Size-based cache management
    - Cache index persistence

    Attributes:
        cache_base_dir: Base directory for cache storage
        max_entries: Maximum number of cache entries to keep
        max_size_bytes: Maximum total cache size in bytes
        console: Console object for shell operations
        index_file: Path to cache index JSON file
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_entries: int = 5,
        max_size_gb: float = 100.0,
        console: Optional[Console] = None
    ):
        """Initialize cache manager.

        Args:
            cache_dir: Base directory for cache (default: ~/.madengine/cache)
            max_entries: Maximum number of cache entries (default: 5)
            max_size_gb: Maximum total cache size in GB (default: 100.0)
            console: Console object for shell operations
        """
        # Set cache directory
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), '.madengine', 'cache')
        self.cache_base_dir = Path(cache_dir)

        # Create cache directory if it doesn't exist
        self.cache_base_dir.mkdir(parents=True, exist_ok=True)

        # Set limits
        self.max_entries = max_entries
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)  # Convert GB to bytes

        # Initialize console
        self.console = console if console is not None else Console()

        # Cache index file
        self.index_file = self.cache_base_dir / 'cache_index.json'

        # Load or create index
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """Load existing cache index or create new one."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    self._cache_index = {k: CacheEntry.from_dict(v) for k, v in data.items()}
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load cache index: {e}. Creating new index.")
                self._cache_index = {}
                self._save_index()
        else:
            self._cache_index = {}
            self._save_index()

    def _save_index(self) -> None:
        """Save cache index to disk."""
        # Use file locking to prevent concurrent writes
        lock_file = self.cache_base_dir / 'cache_index.lock'
        with open(lock_file, 'w') as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                data = {k: v.to_dict() for k, v in self._cache_index.items()}
                with open(self.index_file, 'w') as f:
                    json.dump(data, f, indent=2)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def generate_cache_key(
        self,
        docker_image: str,
        model_tags: List[str],
        additional_context: Dict[str, Any],
        dockerfile: str = ""
    ) -> str:
        """Generate cache key from run configuration.

        Args:
            docker_image: Docker image name
            model_tags: Model tags
            additional_context: Additional context dictionary
            dockerfile: Dockerfile path

        Returns:
            SHA256 hash as cache key
        """
        # Sort and serialize for consistent hashing
        key_components = [
            docker_image,
            '|'.join(sorted(model_tags)),
            json.dumps(additional_context, sort_keys=True),
            dockerfile
        ]
        key_string = '||'.join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_cache_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """Get cache entry by key.

        Args:
            cache_key: Cache key to lookup

        Returns:
            CacheEntry if found, None otherwise
        """
        entry = self._cache_index.get(cache_key)
        if entry:
            # Verify cache directory still exists
            if not Path(entry.cache_dir).exists():
                print(f"Warning: Cache directory {entry.cache_dir} not found. Removing entry.")
                del self._cache_index[cache_key]
                self._save_index()
                return None

            # Update last used
            entry.update_last_used()
            self._save_index()
            return entry
        return None

    def save_cache_entry(
        self,
        docker_image: str,
        docker_tar_path: str,
        run_config: Dict[str, Any],
        model_tags: List[str],
        node_count: int = 1
    ) -> CacheEntry:
        """Save a new cache entry.

        Args:
            docker_image: Docker image name
            docker_tar_path: Path to docker tar file
            run_config: Run configuration dictionary
            model_tags: Model tags
            node_count: Number of nodes for this run

        Returns:
            Created CacheEntry
        """
        # Generate cache key
        cache_key = self.generate_cache_key(
            docker_image=docker_image,
            model_tags=model_tags,
            additional_context=run_config.get('additional_context', {}),
            dockerfile=run_config.get('dockerfile', '')
        )

        # Check if entry already exists
        existing_entry = self._cache_index.get(cache_key)
        if existing_entry:
            existing_entry.update_last_used()
            self._save_index()
            return existing_entry

        # Generate unique cache ID
        cache_id = f"run_{int(time.time())}_{hashlib.md5(cache_key.encode()).hexdigest()[:8]}"

        # Create cache directory
        cache_dir = self.cache_base_dir / cache_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Move docker tar to cache directory
        tar_filename = Path(docker_tar_path).name
        cached_tar_path = cache_dir / tar_filename
        shutil.copy2(docker_tar_path, cached_tar_path)

        # Save run configuration
        config_path = cache_dir / 'run_config.json'
        with open(config_path, 'w') as f:
            json.dump(run_config, f, indent=2)

        # Calculate size
        size_bytes = self._calculate_directory_size(cache_dir)

        # Create cache entry
        now = datetime.now().isoformat()
        entry = CacheEntry(
            cache_id=cache_id,
            cache_key=cache_key,
            docker_image=docker_image,
            docker_tar_path=str(cached_tar_path),
            run_config=run_config,
            model_tags=model_tags,
            created_at=now,
            last_used=now,
            size_bytes=size_bytes,
            node_count=node_count,
            cache_dir=str(cache_dir)
        )

        # Add to index
        self._cache_index[cache_key] = entry
        self._save_index()

        # Evict old entries if needed
        self._evict_if_needed()

        return entry

    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                if filepath.exists():
                    total_size += filepath.stat().st_size
        return total_size

    def _get_total_cache_size(self) -> int:
        """Get total size of all cache entries in bytes."""
        return sum(entry.size_bytes for entry in self._cache_index.values())

    def _evict_if_needed(self) -> None:
        """Evict old entries if limits are exceeded."""
        # Sort entries by last_used (oldest first)
        sorted_entries = sorted(
            self._cache_index.items(),
            key=lambda x: x[1].last_used
        )

        # Evict by count limit
        while len(self._cache_index) > self.max_entries:
            cache_key, entry = sorted_entries.pop(0)
            self._evict_entry(cache_key, entry)
            print(f"Evicted cache entry {entry.cache_id} (count limit)")

        # Evict by size limit
        while self._get_total_cache_size() > self.max_size_bytes and sorted_entries:
            cache_key, entry = sorted_entries.pop(0)
            self._evict_entry(cache_key, entry)
            print(f"Evicted cache entry {entry.cache_id} (size limit)")

    def _evict_entry(self, cache_key: str, entry: CacheEntry) -> None:
        """Evict a single cache entry.

        Args:
            cache_key: Cache key to evict
            entry: CacheEntry to evict
        """
        # Remove from index
        if cache_key in self._cache_index:
            del self._cache_index[cache_key]

        # Remove directory
        cache_dir = Path(entry.cache_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        self._save_index()

    def list_cache_entries(self) -> List[CacheEntry]:
        """List all cache entries sorted by last used (newest first).

        Returns:
            List of CacheEntry objects
        """
        return sorted(
            self._cache_index.values(),
            key=lambda x: x.last_used,
            reverse=True
        )

    def get_cache_entry_by_id(self, cache_id: str) -> Optional[CacheEntry]:
        """Get cache entry by cache ID.

        Args:
            cache_id: Cache ID to lookup

        Returns:
            CacheEntry if found, None otherwise
        """
        for entry in self._cache_index.values():
            if entry.cache_id == cache_id:
                return entry
        return None

    def clear_cache(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache_index)

        # Remove all cache directories
        for entry in self._cache_index.values():
            cache_dir = Path(entry.cache_dir)
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)

        # Clear index
        self._cache_index = {}
        self._save_index()

        return count

    def evict_cache_entry(self, cache_id: str) -> bool:
        """Evict a specific cache entry by ID.

        Args:
            cache_id: Cache ID to evict

        Returns:
            True if evicted, False if not found
        """
        # Find entry by ID
        cache_key = None
        entry = None
        for key, e in self._cache_index.items():
            if e.cache_id == cache_id:
                cache_key = key
                entry = e
                break

        if entry:
            self._evict_entry(cache_key, entry)
            return True
        return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_size = self._get_total_cache_size()
        return {
            'total_entries': len(self._cache_index),
            'max_entries': self.max_entries,
            'total_size_bytes': total_size,
            'total_size_gb': total_size / (1024 ** 3),
            'max_size_gb': self.max_size_bytes / (1024 ** 3),
            'cache_directory': str(self.cache_base_dir),
            'usage_percent': (total_size / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0
        }
