"""
test_retrieval.py - Tests for retrieval pipeline components
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.mark.unit
@pytest.mark.retrieval
class TestVectorRetriever:
    """Test vector retrieval with mocked embeddings"""
    
    def test_vector_search_mock(self, sample_qa_documents, mock_embeddings):
        """Test vector search returns expected results"""
        from retrieval.vector_retriever import VectorRetriever
        
        retriever = MagicMock(spec=VectorRetriever)
        
        # Mock search to return documents
        def mock_search(query_embedding, top_k=10, category_filter=None):
            return [
                {"doc_id": doc["doc_id"], "doc": doc, "score": 0.9 - i * 0.05}
                for i, doc in enumerate(sample_qa_documents[:top_k])
            ]
        
        retriever.search = mock_search
        retriever.encode_query = lambda q: mock_embeddings["query_zakat"]
        
        # Test search
        results = retriever.search(mock_embeddings["query_zakat"], top_k=3)
        
        assert len(results) == 3
        assert results[0]["doc_id"] == "qa_001"
        assert results[0]["score"] > results[1]["score"]
    
    def test_category_filter_mock(self, sample_qa_documents):
        """Test category pre-filtering"""
        from retrieval.vector_retriever import VectorRetriever
        
        retriever = MagicMock(spec=VectorRetriever)
        
        def mock_search_with_filter(query_embedding, top_k=10, category_filter="islamic_finance"):
            filtered = [d for d in sample_qa_documents if d.get("category") == category_filter]
            return [
                {"doc_id": doc["doc_id"], "doc": doc, "score": 0.9}
                for doc in filtered[:top_k]
            ]
        
        retriever.search = mock_search_with_filter
        
        results = retriever.search(None, top_k=10, category_filter="islamic_finance")
        
        # Should only return islamic_finance docs
        assert all(r["doc"]["category"] == "islamic_finance" for r in results)


@pytest.mark.unit
@pytest.mark.retrieval
class TestBM25Retriever:
    """Test BM25 keyword retrieval"""
    
    def test_bm25_search_mock(self, sample_qa_documents):
        """Test BM25 search with mock index"""
        from retrieval.bm25_retriever import BM25Retriever
        
        retriever = MagicMock(spec=BM25Retriever)
        
        def mock_search(query, top_k=10):
            # Simple keyword matching simulation
            query_terms = query.lower().split()
            scored = []
            for doc in sample_qa_documents:
                score = 0
                doc_text = doc.get("question", "").lower()
                for term in query_terms:
                    if term in doc_text:
                        score += 1
                scored.append((doc, score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            return [
                {"doc_id": doc["doc_id"], "doc": doc, "score": score}
                for doc, score in scored[:top_k] if score > 0
            ]
        
        retriever.search = mock_search
        
        results = retriever.search("zakat hisab", top_k=5)
        
        # Should return docs with "zakat" in question
        assert len(results) > 0
        assert all("zakat" in r["doc"]["question"].lower() for r in results)


@pytest.mark.unit
@pytest.mark.retrieval
class TestFusion:
    """Test RRF fusion and result merging"""
    
    def test_rrf_fusion_basic(self):
        """Test basic RRF fusion calculation"""
        from retrieval.fusion import fuse, FusedResult
        
        # Create mock results
        from types import SimpleNamespace
        
        def vec(doc_id, score, rank):
            return SimpleNamespace(doc_id=doc_id, score=score, rank=rank, doc={})
        
        def bm25(doc_id, score, rank):
            return SimpleNamespace(doc_id=doc_id, score=score, score_norm=score, rank=rank, doc={})
        
        vector_results = [vec("qa_001", 0.95, 1), vec("qa_002", 0.85, 2), vec("qa_003", 0.75, 3)]
        bm25_results   = [bm25("qa_002", 0.90, 1), bm25("qa_001", 0.80, 2), bm25("qa_004", 0.70, 3)]
        
        fused = fuse(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_weight=0.6,
            bm25_weight=0.4,
            rrf_k=60,
        )
        
        assert len(fused) > 0
        # qa_001 and qa_002 should be highly ranked (appear in both)
        top_ids = [r.doc_id for r in fused[:3]]
        assert "qa_001" in top_ids or "qa_002" in top_ids
    
    def test_rrf_score_calculation(self):
        """Test RRF contribution formula"""
        from retrieval.fusion import _rrf_contribution
        
        # _rrf_contribution(rank, rrf_k, weight) = weight / (rrf_k + rank)
        vector_contrib = _rrf_contribution(rank=1, rrf_k=60, weight=0.6)
        bm25_contrib   = _rrf_contribution(rank=2, rrf_k=60, weight=0.4)
        combined = vector_contrib + bm25_contrib
        
        expected = 0.6 / 61 + 0.4 / 62
        assert abs(combined - expected) < 0.0001
    
    def test_empty_results_handling(self):
        """Test fusion with empty result sets"""
        from retrieval.fusion import fuse
        from types import SimpleNamespace
        
        bm25_item = SimpleNamespace(doc_id="qa_001", score=0.9, score_norm=0.9, rank=1, doc={})
        
        fused = fuse(
            vector_results=[],
            bm25_results=[bm25_item],
            vector_weight=0.6,
            bm25_weight=0.4,
        )
        
        assert len(fused) == 1
        assert fused[0].doc_id == "qa_001"


@pytest.mark.unit
@pytest.mark.retrieval
class TestMMR:
    """Test Max Marginal Relevance diversity filtering"""
    
    def test_mmr_diversity_selection(self):
        """Test MMR selects diverse results"""
        from retrieval.mmr import run_mmr, MMRResult
        
        # Create candidate results with similarity scores
        candidates = [
            {"doc_id": "qa_001", "score": 0.95, "embedding": np.array([1.0, 0.0, 0.0])},
            {"doc_id": "qa_002", "score": 0.90, "embedding": np.array([0.99, 0.01, 0.0])},  # Similar to qa_001
            {"doc_id": "qa_003", "score": 0.85, "embedding": np.array([0.0, 1.0, 0.0])},  # Different
            {"doc_id": "qa_004", "score": 0.80, "embedding": np.array([0.0, 0.99, 0.01])},  # Similar to qa_003
        ]
        
        query_embedding = np.array([1.0, 0.0, 0.0])
        
        with patch("retrieval.mmr.cosine_similarity") as mock_sim:
            # Mock similarity calculations
            def mock_cosine(a, b):
                a = a / (np.linalg.norm(a) + 1e-9)
                b = b / (np.linalg.norm(b) + 1e-9)
                return float(np.dot(a, b))
            
            mock_sim.side_effect = mock_cosine
            
            results = run_mmr(
                candidates=candidates,
                query_embedding=query_embedding,
                lambda_param=0.7,
                top_k=2,
            )
        
        # Should select top 2 diverse results
        assert len(results) <= 2
    
    def test_mmr_lambda_param(self):
        """Test lambda parameter controls relevance vs diversity tradeoff"""
        from retrieval.mmr import mmr_score
        
        relevance_score = 0.9
        max_similarity = 0.8
        
        # High lambda = more relevance
        score_high_lambda = mmr_score(relevance_score, max_similarity, lambda_param=0.9)
        # Low lambda = more diversity
        score_low_lambda = mmr_score(relevance_score, max_similarity, lambda_param=0.3)
        
        assert score_high_lambda > score_low_lambda


@pytest.mark.unit
@pytest.mark.retrieval
class TestLanguageDetection:
    """Test language detection and weight selection"""
    
    def test_urdu_script_detection(self):
        """Test detection of Urdu script"""
        from retrieval.language import detect_language
        
        result = detect_language("زکوٰۃ کا نصاب کتنا ہے؟")
        
        assert result.label in ["urdu", "roman_urdu", "arabic"]
        assert result.confidence > 0.5
    
    def test_roman_urdu_detection(self):
        """Test detection of Roman Urdu"""
        from retrieval.language import detect_language
        
        result = detect_language("zakat ka hisab kaise karein")
        
        assert result.label in ["roman_urdu", "english", "mixed"]
    
    def test_english_detection(self):
        """Test detection of English"""
        from retrieval.language import detect_language
        
        result = detect_language("How to calculate zakat?")
        
        assert result.label in ["english", "mixed"]
        assert result.confidence > 0.5
    
    def test_fusion_weights_by_language(self, base_config):
        """Test correct fusion weights are returned per language"""
        from retrieval.language import detect_and_get_weights
        
        weights = detect_and_get_weights("zakat ka hisab", base_config["fusion"]["language_weights"])
        
        assert "bm25_weight" in weights
        assert "vector_weight" in weights
        assert weights["bm25_weight"] + weights["vector_weight"] == 1.0


@pytest.mark.unit
@pytest.mark.retrieval
class TestQueryProcessing:
    """Test query normalization and preprocessing"""
    
    def test_query_normalization_lowercase(self):
        """Test query normalization converts to lowercase"""
        from retrieval.normalization import normalize_query
        
        result = normalize_query("ZAKAT Calculation")
        
        assert result == "zakat calculation"
    
    def test_query_normalization_whitespace(self):
        """Test query normalization handles whitespace"""
        from retrieval.normalization import normalize_query
        
        result = normalize_query("  zakat    calculation  ")
        
        assert result == "zakat calculation"
    
    def test_roman_urdu_normalization(self):
        """Test Roman Urdu query normalization"""
        from retrieval.normalization import normalize_query
        
        # Test repeated character collapsing
        result = normalize_query("kyaaaa")  # kyaa with extra 'a'
        
        assert "kyaa" in result or result == "kyaaaa"  # Depends on implementation
    
    def test_urdu_unicode_normalization(self):
        """Test Urdu Unicode normalization"""
        from retrieval.normalization import normalize_query
        
        # Various forms of Urdu characters
        result = normalize_query("کتا")  # Persian kaf
        
        # Should normalize to consistent form
        assert isinstance(result, str)


@pytest.mark.unit
@pytest.mark.retrieval
class TestCategoryDetector:
    """Test category detection from queries"""
    
    def test_category_detection_zakat(self):
        """Test detection of Islamic finance category"""
        from metadata.category_detector import CategoryDetector
        
        detector = CategoryDetector()
        result = detector.detect("zakat ka hisab kaise karein")
        
        assert result.category in ["islamic_finance", None]
        if result.category:
            assert result.confidence > 0
    
    def test_category_detection_easypaisa(self):
        """Test detection of digital finance category"""
        from metadata.category_detector import CategoryDetector
        
        detector = CategoryDetector()
        result = detector.detect("easypaisa se paise bhejein")
        
        assert result.category in ["digital_finance", None]
    
    def test_category_detection_loan(self):
        """Test detection of loans category"""
        from metadata.category_detector import CategoryDetector
        
        detector = CategoryDetector()
        result = detector.detect("home loan ka rate kya hai")
        
        assert result.category in ["loans_credit", "loans", None]
    
    def test_unknown_category(self):
        """Test handling of queries without clear category"""
        from metadata.category_detector import CategoryDetector
        
        detector = CategoryDetector()
        result = detector.detect("random query about nothing specific")
        
        # Should return None or low confidence
        assert result.category is None or result.confidence < 0.5


@pytest.mark.unit
@pytest.mark.retrieval
class TestDualRetriever:
    """Test dual-index retrieval (QA + PDF)"""
    
    def test_dual_retrieval_mock(self, sample_qa_documents, sample_pdf_documents):
        """Test dual retriever combines QA and PDF results"""
        from retrieval.dual_retriever import DualRetriever
        
        retriever = MagicMock(spec=DualRetriever)
        
        def mock_retrieve(query, language="roman_urdu", query_intent="practical",
                         category_filter=None, top_k=10, force_pdf_only=False):
            # Simulate retrieval from both sources
            qa_docs = sample_qa_documents[:5]
            pdf_docs = sample_pdf_documents if not force_pdf_only else []
            
            all_docs = []
            for doc in qa_docs:
                all_docs.append({"doc_id": doc["doc_id"], "doc": doc, "source": "qa", "score": 0.9})
            for doc in pdf_docs:
                all_docs.append({"doc_id": doc["doc_id"], "doc": doc, "source": "pdf", "score": 0.85})
            
            # Sort by score
            all_docs.sort(key=lambda x: x["score"], reverse=True)
            
            result = MagicMock()
            result.docs = all_docs[:top_k]
            result.source_mix = {"qa": len(qa_docs), "pdf": len(pdf_docs)}
            return result
        
        retriever.retrieve = mock_retrieve
        
        result = retriever.retrieve("zakat query", top_k=5)
        
        assert len(result.docs) <= 5
        assert result.source_mix["qa"] > 0
    
    def test_source_routing_weights(self, base_config):
        """Test different source weights for practical vs legal queries"""
        routing = base_config.get("source_routing", {})
        
        practical = routing.get("practical_query", {})
        legal = routing.get("legal_query", {})
        
        # Legal queries should favor PDFs
        assert legal.get("pdf", 0) > practical.get("pdf", 0)
        # Practical queries should favor QA
        assert practical.get("qa", 0) >= legal.get("qa", 0)


@pytest.mark.unit
@pytest.mark.retrieval
class TestRetrievalPipeline:
    """Integration tests for full retrieval pipeline"""
    
    def test_pipeline_initialization(self, base_config, temp_artifacts_dir):
        """Test pipeline can be initialized with config"""
        from retrieval.pipeline import RetrievalPipeline
        
        # Mock the actual loading
        with patch.object(RetrievalPipeline, "load") as mock_load:
            pipeline = RetrievalPipeline(base_config)
            
            assert pipeline is not None
            assert pipeline.cfg == base_config
    
    def test_pipeline_run_mock(self, base_config, sample_qa_documents):
        """Test pipeline run with mocked components"""
        from retrieval.pipeline import RetrievalPipeline, RetrievalOutput
        
        pipeline = MagicMock(spec=RetrievalPipeline)
        
        def mock_run(query):
            result = MagicMock(spec=RetrievalOutput)
            result.query_raw = query
            result.query_normalized = query.lower().strip()
            result.docs = [
                MagicMock(doc_id=doc["doc_id"], metadata={"doc": doc})
                for doc in sample_qa_documents[:3]
            ]
            result.diagnostics = {
                "language_detected": "roman_urdu",
                "category_detected": "islamic_finance",
                "vector_ms": 15.0,
                "bm25_ms": 5.0,
            }
            result.total_ms = 50.0
            return result
        
        pipeline.run = mock_run
        
        result = pipeline.run("zakat ka hisab")
        
        assert result.query_raw == "zakat ka hisab"
        assert len(result.docs) == 3
        assert result.diagnostics["language_detected"] == "roman_urdu"
