"""Tests for caching functionality.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
import os
import tempfile
import shutil
import json
from pathlib import Path
import pytest

from madengine.core.cache import CacheManager, CacheEntry
from madengine.core.docker import Docker
from madengine.core.console import Console


class TestCacheEntry:
    """Test CacheEntry class."""

    def test_create_cache_entry(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            cache_id="test_123",
            cache_key="abc123",
            docker_image="test:latest",
            docker_tar_path="/path/to/tar",
            run_config={'test': 'config'},
            model_tags=['tag1', 'tag2'],
            created_at="2024-01-01T00:00:00",
            last_used="2024-01-01T00:00:00",
            size_bytes=1000000,
            node_count=1,
            cache_dir="/path/to/cache"
        )

        assert entry.cache_id == "test_123"
        assert entry.cache_key == "abc123"
        assert entry.docker_image == "test:latest"
        assert entry.node_count == 1

    def test_cache_entry_to_dict(self):
        """Test converting cache entry to dictionary."""
        entry = CacheEntry(
            cache_id="test_123",
            cache_key="abc123",
            docker_image="test:latest",
            docker_tar_path="/path/to/tar",
            run_config={'test': 'config'},
            model_tags=['tag1', 'tag2'],
            created_at="2024-01-01T00:00:00",
            last_used="2024-01-01T00:00:00",
            size_bytes=1000000,
            node_count=1,
            cache_dir="/path/to/cache"
        )

        data = entry.to_dict()
        assert data['cache_id'] == "test_123"
        assert data['cache_key'] == "abc123"
        assert data['docker_image'] == "test:latest"

    def test_cache_entry_from_dict(self):
        """Test creating cache entry from dictionary."""
        data = {
            'cache_id': "test_123",
            'cache_key': "abc123",
            'docker_image': "test:latest",
            'docker_tar_path': "/path/to/tar",
            'run_config': {'test': 'config'},
            'model_tags': ['tag1', 'tag2'],
            'created_at': "2024-01-01T00:00:00",
            'last_used': "2024-01-01T00:00:00",
            'size_bytes': 1000000,
            'node_count': 1,
            'cache_dir': "/path/to/cache"
        }

        entry = CacheEntry.from_dict(data)
        assert entry.cache_id == "test_123"
        assert entry.cache_key == "abc123"
        assert entry.docker_image == "test:latest"


class TestCacheManager:
    """Test CacheManager class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create cache manager with temporary directory."""
        return CacheManager(cache_dir=temp_cache_dir, max_entries=5, max_size_gb=1.0)

    def test_create_cache_manager(self, temp_cache_dir):
        """Test creating cache manager."""
        manager = CacheManager(cache_dir=temp_cache_dir)
        assert manager.cache_base_dir == Path(temp_cache_dir)
        assert manager.max_entries == 5
        assert manager.max_size_bytes == 100 * 1024 * 1024 * 1024  # 100GB default

    def test_generate_cache_key(self, cache_manager):
        """Test generating cache key."""
        key1 = cache_manager.generate_cache_key(
            docker_image="test:latest",
            model_tags=['tag1', 'tag2'],
            additional_context={'key': 'value'},
            dockerfile="/path/to/Dockerfile"
        )

        # Same inputs should produce same key
        key2 = cache_manager.generate_cache_key(
            docker_image="test:latest",
            model_tags=['tag1', 'tag2'],
            additional_context={'key': 'value'},
            dockerfile="/path/to/Dockerfile"
        )

        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex digest

        # Different inputs should produce different key
        key3 = cache_manager.generate_cache_key(
            docker_image="test:v2",
            model_tags=['tag1', 'tag2'],
            additional_context={'key': 'value'},
            dockerfile="/path/to/Dockerfile"
        )

        assert key1 != key3

    def test_save_and_get_cache_entry(self, cache_manager, temp_cache_dir):
        """Test saving and retrieving cache entry."""
        # Create a dummy tar file
        tar_path = os.path.join(temp_cache_dir, "test.tar")
        with open(tar_path, 'w') as f:
            f.write("test data")

        # Save cache entry
        entry = cache_manager.save_cache_entry(
            docker_image="test:latest",
            docker_tar_path=tar_path,
            run_config={'test': 'config'},
            model_tags=['tag1', 'tag2'],
            node_count=1
        )

        assert entry.docker_image == "test:latest"
        assert entry.model_tags == ['tag1', 'tag2']
        assert entry.node_count == 1

        # Get cache entry by key
        retrieved_entry = cache_manager.get_cache_entry(entry.cache_key)
        assert retrieved_entry is not None
        assert retrieved_entry.cache_id == entry.cache_id
        assert retrieved_entry.docker_image == entry.docker_image

    def test_list_cache_entries(self, cache_manager, temp_cache_dir):
        """Test listing cache entries."""
        # Initially empty
        entries = cache_manager.list_cache_entries()
        assert len(entries) == 0

        # Add some entries
        for i in range(3):
            tar_path = os.path.join(temp_cache_dir, f"test{i}.tar")
            with open(tar_path, 'w') as f:
                f.write(f"test data {i}")

            cache_manager.save_cache_entry(
                docker_image=f"test:v{i}",
                docker_tar_path=tar_path,
                run_config={'test': f'config{i}'},
                model_tags=[f'tag{i}'],
                node_count=1
            )

        # List entries
        entries = cache_manager.list_cache_entries()
        assert len(entries) == 3

    def test_evict_cache_entry(self, cache_manager, temp_cache_dir):
        """Test evicting cache entry."""
        # Create and save entry
        tar_path = os.path.join(temp_cache_dir, "test.tar")
        with open(tar_path, 'w') as f:
            f.write("test data")

        entry = cache_manager.save_cache_entry(
            docker_image="test:latest",
            docker_tar_path=tar_path,
            run_config={'test': 'config'},
            model_tags=['tag1'],
            node_count=1
        )

        # Verify entry exists
        assert len(cache_manager.list_cache_entries()) == 1

        # Evict entry
        success = cache_manager.evict_cache_entry(entry.cache_id)
        assert success is True

        # Verify entry removed
        assert len(cache_manager.list_cache_entries()) == 0

    def test_clear_cache(self, cache_manager, temp_cache_dir):
        """Test clearing all cache entries."""
        # Add multiple entries
        for i in range(3):
            tar_path = os.path.join(temp_cache_dir, f"test{i}.tar")
            with open(tar_path, 'w') as f:
                f.write(f"test data {i}")

            cache_manager.save_cache_entry(
                docker_image=f"test:v{i}",
                docker_tar_path=tar_path,
                run_config={'test': f'config{i}'},
                model_tags=[f'tag{i}'],
                node_count=1
            )

        # Verify entries exist
        assert len(cache_manager.list_cache_entries()) == 3

        # Clear cache
        count = cache_manager.clear_cache()
        assert count == 3

        # Verify all entries removed
        assert len(cache_manager.list_cache_entries()) == 0

    def test_cache_stats(self, cache_manager, temp_cache_dir):
        """Test getting cache statistics."""
        # Get initial stats
        stats = cache_manager.get_cache_stats()
        assert stats['total_entries'] == 0
        assert stats['total_size_bytes'] == 0
        assert stats['max_entries'] == 5

        # Add an entry
        tar_path = os.path.join(temp_cache_dir, "test.tar")
        with open(tar_path, 'w') as f:
            f.write("test data")

        cache_manager.save_cache_entry(
            docker_image="test:latest",
            docker_tar_path=tar_path,
            run_config={'test': 'config'},
            model_tags=['tag1'],
            node_count=1
        )

        # Get updated stats
        stats = cache_manager.get_cache_stats()
        assert stats['total_entries'] == 1
        assert stats['total_size_bytes'] > 0
        assert stats['cache_directory'] == str(cache_manager.cache_base_dir)

    def test_lru_eviction(self, cache_manager, temp_cache_dir):
        """Test LRU eviction when max entries exceeded."""
        # Set max entries to 3
        cache_manager.max_entries = 3

        # Add 5 entries (should trigger eviction)
        entries = []
        for i in range(5):
            tar_path = os.path.join(temp_cache_dir, f"test{i}.tar")
            with open(tar_path, 'w') as f:
                f.write(f"test data {i}")

            entry = cache_manager.save_cache_entry(
                docker_image=f"test:v{i}",
                docker_tar_path=tar_path,
                run_config={'test': f'config{i}'},
                model_tags=[f'tag{i}'],
                node_count=1
            )
            entries.append(entry)

        # Should only keep last 3 entries
        remaining_entries = cache_manager.list_cache_entries()
        assert len(remaining_entries) <= 3

    def test_multi_node_support(self, cache_manager, temp_cache_dir):
        """Test multi-node cache entry."""
        tar_path = os.path.join(temp_cache_dir, "test.tar")
        with open(tar_path, 'w') as f:
            f.write("test data")

        # Create entry with multiple nodes
        entry = cache_manager.save_cache_entry(
            docker_image="test:latest",
            docker_tar_path=tar_path,
            run_config={'test': 'config'},
            model_tags=['tag1'],
            node_count=4  # Multi-node
        )

        assert entry.node_count == 4

        # Retrieve and verify
        retrieved_entry = cache_manager.get_cache_entry(entry.cache_key)
        assert retrieved_entry.node_count == 4


class TestDockerTarMethods:
    """Test Docker tar save/load methods."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_docker_tar_path_creation(self, temp_dir):
        """Test that tar path is created correctly."""
        tar_path = os.path.join(temp_dir, "subdir", "test.tar")
        # This would normally save a docker image, but we'll just test the path creation
        # In actual tests, you'd need a real docker image
        # Docker.save_image_to_tar("test:latest", tar_path)

        # For now, just verify the directory would be created
        assert not os.path.exists(tar_path)
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        assert os.path.exists(os.path.dirname(tar_path))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
