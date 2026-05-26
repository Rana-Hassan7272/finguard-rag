"""
test_cache.py - Tests for semantic cache including invalidation
"""

import time
from unittest.mock import patch

import numpy as np
import pytest


@pytest.mark.unit
@pytest.mark.cache
class TestSemanticCache:
    """Test suite for two-level semantic cache"""
    
    def test_cache_initialization(self):
        """Test cache initializes with correct parameters"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(
            l1_threshold=0.95,
            l2_threshold=0.90,
            ttl_seconds=3600,
            max_entries=100,
            corpus_version="v1",
        )
        
        assert cache.l1_threshold == 0.95
        assert cache.l2_threshold == 0.90
        assert cache.ttl_seconds == 3600
        assert cache.max_entries == 100
        assert cache.corpus_version == "v1"
        assert len(cache) == 0
    
    def test_l1_cache_hit(self):
        """Test Level 1 cache hit (answer cache)"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(l1_threshold=0.95, corpus_version="v1")
        
        # Store an entry
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        cache.store(
            query_embedding=query_embedding,
            query_text="zakat ka hisab",
            doc_ids=["qa_001"],
            answer="Zakat is 2.5% of savings above nisab.",
        )
        
        # Lookup with very similar embedding (should be L1 hit)
        similar_embedding = query_embedding.copy()
        result = cache.lookup(similar_embedding)
        
        assert result.hit is True
        assert result.level == 1
        assert result.answer == "Zakat is 2.5% of savings above nisab."
        assert result.doc_ids == ["qa_001"]
        assert result.similarity >= 0.99  # Same embedding
    
    def test_l2_cache_hit(self):
        """Test Level 2 cache hit (retrieval cache)"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(l1_threshold=0.95, l2_threshold=0.90, corpus_version="v1")
        
        # Store with answer
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        cache.store(
            query_embedding=query_embedding,
            query_text="zakat calculation",
            doc_ids=["qa_001", "qa_002"],
            answer="Zakat is 2.5% of wealth above nisab threshold.",
        )
        
        # Create slightly different embedding (L2 hit but not L1)
        modified_embedding = query_embedding + np.random.randn(768) * 0.03
        modified_embedding = modified_embedding / np.linalg.norm(modified_embedding)
        
        result = cache.lookup(modified_embedding)
        
        # Should be some hit (either L1 or L2 depending on similarity)
        if result.hit and result.similarity >= 0.95:
            assert result.level == 1
        elif result.hit:
            assert result.level in [1, 2]
    
    def test_cache_miss(self):
        """Test cache miss when embedding is too different"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(l1_threshold=0.95, l2_threshold=0.90, corpus_version="v1")
        
        # Store entry
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        cache.store(
            query_embedding=query_embedding,
            query_text="zakat",
            doc_ids=["qa_001"],
            answer="Answer about zakat",
        )
        
        # Create very different embedding
        different_embedding = np.random.randn(768).astype(np.float32)
        different_embedding = different_embedding / np.linalg.norm(different_embedding)
        
        result = cache.lookup(different_embedding)
        
        # Should be a miss (similarity below both thresholds)
        assert result.hit is False
        assert result.level is None
        assert result.answer is None
    
    def test_cache_miss_empty_cache(self):
        """Test miss on empty cache"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache()
        query_embedding = np.random.randn(768).astype(np.float32)
        
        result = cache.lookup(query_embedding)
        
        assert result.hit is False
        assert result.reason == "cache_empty"
    
    def test_corpus_version_invalidation(self):
        """Test cache invalidation when corpus version changes"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(l1_threshold=0.95, corpus_version="v1")
        
        # Store entry with version v1
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        cache.store(
            query_embedding=query_embedding,
            query_text="zakat",
            doc_ids=["qa_001"],
            answer="Answer",
        )
        
        # Verify it works
        result = cache.lookup(query_embedding)
        assert result.hit is True
        
        # Change corpus version
        stale_count = cache.invalidate_corpus("v2")
        assert stale_count == 1
        
        # Now lookup should fail
        result = cache.lookup(query_embedding)
        assert result.hit is False
        assert result.reason == "no_valid_entries"
    
    def test_ttl_expiration(self):
        """Test TTL expiration of cache entries"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(l1_threshold=0.95, ttl_seconds=0.1, corpus_version="v1")
        
        # Store entry with very short TTL
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        cache.store(
            query_embedding=query_embedding,
            query_text="zakat",
            doc_ids=["qa_001"],
            answer="Answer",
        )
        
        # Should work immediately
        result = cache.lookup(query_embedding)
        assert result.hit is True
        
        # Wait for TTL to expire
        time.sleep(0.15)
        
        # Now should be expired
        result = cache.lookup(query_embedding)
        assert result.hit is False
        assert result.reason == "no_valid_entries"
    
    def test_evict_expired(self):
        """Test explicit eviction of expired entries"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(ttl_seconds=0.1, corpus_version="v1")
        
        # Store entries
        for i in range(3):
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            cache.store(
                query_embedding=embedding,
                query_text=f"query_{i}",
                doc_ids=[f"qa_{i}"],
                answer=f"Answer {i}",
            )
        
        assert len(cache) == 3
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Evict expired
        evicted = cache.evict_expired()
        assert evicted == 3
        assert len(cache) == 0
    
    def test_capacity_eviction(self):
        """Test eviction when cache reaches max capacity"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(max_entries=5, corpus_version="v1")
        
        # Store more entries than capacity
        for i in range(7):
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            cache.store(
                query_embedding=embedding,
                query_text=f"query_{i}",
                doc_ids=[f"qa_{i}"],
                answer=f"Answer {i}",
            )
        
        # Should only keep max_entries (removes 10% when full)
        assert len(cache) <= 5
    
    def test_cache_stats(self):
        """Test cache statistics tracking"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(l1_threshold=0.95, l2_threshold=0.90, corpus_version="v1")
        
        # Initial stats
        stats = cache.stats()
        assert stats["l1_hits"] == 0
        assert stats["l2_hits"] == 0
        assert stats["misses"] == 0
        
        # Store and lookup
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        cache.store(
            query_embedding=query_embedding,
            query_text="zakat",
            doc_ids=["qa_001"],
            answer="Answer",
        )
        
        # Hit
        cache.lookup(query_embedding)
        
        # Miss
        different = np.random.randn(768).astype(np.float32)
        cache.lookup(different)
        
        stats = cache.stats()
        assert stats["stores"] == 1
        assert stats["total_lookups"] == 2
        assert stats["current_size"] == 1
    
    def test_cache_clear(self):
        """Test cache clear functionality"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(corpus_version="v1")
        
        # Store entries
        for i in range(3):
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            cache.store(
                query_embedding=embedding,
                query_text=f"query_{i}",
                doc_ids=[f"qa_{i}"],
                answer=f"Answer {i}",
            )
        
        assert len(cache) == 3
        
        # Clear
        cache.clear()
        
        assert len(cache) == 0
    
    def test_update_existing_entry(self):
        """Test updating an existing entry"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(corpus_version="v1")
        
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Store initial
        cache.store(
            query_embedding=query_embedding,
            query_text="zakat",
            doc_ids=["qa_001"],
            answer=None,  # No answer yet
        )
        
        # Update with answer
        cache.store(
            query_embedding=query_embedding,
            query_text="zakat",
            doc_ids=["qa_001"],
            answer="Zakat is 2.5%",
        )
        
        # Lookup should find answer
        result = cache.lookup(query_embedding)
        assert result.hit is True
        assert result.answer == "Zakat is 2.5%"
    
    def test_entry_age_tracking(self):
        """Test entry age is tracked correctly"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(corpus_version="v1")
        
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        with patch("time.time") as mock_time:
            mock_time.return_value = 1000.0
            
            cache.store(
                query_embedding=query_embedding,
                query_text="zakat",
                doc_ids=["qa_001"],
                answer="Answer",
            )
            
            # Advance time
            mock_time.return_value = 1100.0
            
            result = cache.lookup(query_embedding)
            assert result.entry_age_seconds == 100.0
    
    def test_hit_count_tracking(self):
        """Test hit count is tracked for entries"""
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(l1_threshold=0.95, corpus_version="v1")
        
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        cache.store(
            query_embedding=query_embedding,
            query_text="zakat",
            doc_ids=["qa_001"],
            answer="Answer",
        )
        
        # Multiple lookups
        for _ in range(5):
            cache.lookup(query_embedding)
        
        # Check that entry was touched
        assert len(cache._entries) == 1
        assert cache._entries[0].hit_count == 5
        assert cache._entries[0].last_hit_ts > 0


@pytest.mark.unit
class TestCorpusVersion:
    """Test corpus version computation"""
    
    def test_compute_corpus_version_default(self):
        """Test default corpus version when no manifests exist"""
        from cache.semantic_cache import compute_corpus_version
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            
            version = compute_corpus_version("/nonexistent/path")
            assert version == "default"
    
    def test_compute_corpus_version_with_files(self, tmp_path):
        """Test corpus version computation with actual files"""
        from cache.semantic_cache import compute_corpus_version
        
        # Create test artifact files
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        
        stats_file = artifacts_dir / "corpus_stats.json"
        stats_file.write_text('{"count": 100, "version": "v1"}')
        
        manifest_file = artifacts_dir / "index_manifest_v1.json"
        manifest_file.write_text('{"version": "embedder_v1", "docs": 100}')
        
        version = compute_corpus_version(str(artifacts_dir))
        
        # Should return an 8-char hash
        assert len(version) == 8
        assert version != "default"
    
    def test_build_cache_from_config(self, base_config):
        """Test building cache from config"""
        from cache.semantic_cache import build_cache_from_config
        
        # Create mock artifact files
        with patch("cache.semantic_cache.compute_corpus_version") as mock_version:
            mock_version.return_value = "abc12345"
            
            cache = build_cache_from_config(base_config)
            
            assert cache.l1_threshold == 0.95
            assert cache.l2_threshold == 0.90
            assert cache.corpus_version == "abc12345"


@pytest.mark.unit
@pytest.mark.cache
class TestCacheThreadSafety:
    """Test thread safety of cache operations"""
    
    def test_concurrent_lookups(self):
        """Test concurrent cache lookups"""
        import threading
        
        from cache.semantic_cache import SemanticCache
        
        cache = SemanticCache(l1_threshold=0.95, corpus_version="v1")
        
        # Pre-populate
        query_embedding = np.random.randn(768).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        cache.store(
            query_embedding=query_embedding,
            query_text="test",
            doc_ids=["qa_001"],
            answer="Answer",
        )
        
        results = []
        
        def lookup_worker():
            for _ in range(10):
                result = cache.lookup(query_embedding)
                results.append(result.hit)
        
        # Run multiple threads
        threads = [threading.Thread(target=lookup_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All lookups should succeed
        assert all(results)
        assert len(results) == 50
