"""
conftest.py - Pytest configuration and shared fixtures
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml


# =============================================================================
# Path Setup
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    """Add repo root to path for imports"""
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Fixtures - Configuration
# =============================================================================

@pytest.fixture(scope="session")
def base_config():
    """Load actual retrieval config from repository"""
    config_path = REPO_ROOT / "retrieval" / "configs" / "retrieval_config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    # Fallback minimal config for testing
    return {
        "embedding_model": {
            "hub_id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "version_tag": "test_v1",
            "embedding_dim": 768,
            "normalize_embeddings": True,
        },
        "retrieval": {
            "vector_k": 10,
            "bm25_k": 10,
            "hybrid_candidate_k": 20,
            "mmr_output_k": 10,
            "reranker_output_k": 3,
        },
        "faiss": {
            "index_type": "IndexFlatIP",
            "dimension": 768,
        },
        "bm25": {
            "tokenizer_mode": "whitespace",
            "k1": 1.5,
            "b": 0.75,
        },
        "fusion": {
            "method": "rrf",
            "rrf_k": 60,
            "language_weights": {
                "roman_urdu": {"bm25_weight": 0.30, "vector_weight": 0.70},
                "urdu": {"bm25_weight": 0.40, "vector_weight": 0.60},
                "english": {"bm25_weight": 0.45, "vector_weight": 0.55},
            },
        },
        "mmr": {"lambda": 0.7, "candidate_pool_size": 20, "output_k": 10},
        "reranker": {
            "model_id": "BAAI/bge-reranker-base",
            "confidence_threshold": 0.55,
            "extractive_gate_floor": 0.22,
            "pdf_retry_threshold": 0.25,
            "input_k": 6,
            "output_k": 3,
            "batch_size": 16,
            "max_length": 384,
        },
        "cache": {
            "enabled": True,
            "similarity_threshold": 0.92,
            "ttl_seconds": 86400,
            "max_entries": 2000,
            "l1_threshold": 0.95,
            "l2_threshold": 0.90,
        },
        "generation": {
            "primary_provider": "groq",
            "primary_model": "llama-3.3-70b-versatile",
            "fallback_enabled": True,
            "fallback_provider": "openai",
            "fallback_model": "gpt-4o-mini",
            "max_tokens": 300,
            "temperature": 0.1,
            "timeout_seconds": 10,
            "max_retries_primary": 3,
            "max_retries_fallback": 2,
        },
        "artifacts": {"root": "retrieval/artifacts"},
    }


@pytest.fixture
def temp_config_file(base_config, tmp_path):
    """Create a temporary config file for testing"""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(base_config, f)
    return str(config_path)


# =============================================================================
# Fixtures - Mock Data
# =============================================================================

@pytest.fixture
def sample_qa_documents():
    """Sample QA documents for testing"""
    return [
        {
            "doc_id": "qa_001",
            "doc_type": "qa",
            "question": "zakat ka hisab kaise karein",
            "answer": "Zakat 2.5% of total savings and assets above nisab threshold.",
            "category": "islamic_finance",
            "language": "roman_urdu",
        },
        {
            "doc_id": "qa_002",
            "doc_type": "qa",
            "question": "What is the nisab for zakat in Pakistan?",
            "answer": "Nisab is approximately 87.48g gold or 612.36g silver value.",
            "category": "islamic_finance",
            "language": "english",
        },
        {
            "doc_id": "qa_003",
            "doc_type": "qa",
            "question": "easypaisa se paise kaise bhejein",
            "answer": "Open EasyPaisa app, select 'Send Money', enter recipient mobile number.",
            "category": "digital_finance",
            "language": "roman_urdu",
        },
        {
            "doc_id": "qa_004",
            "doc_type": "qa",
            "question": "riba aur interest mein kya farq hai",
            "answer": "Riba is haram in Islam. Interest in conventional banking is similar but from different perspective.",
            "category": "islamic_finance",
            "language": "roman_urdu",
        },
        {
            "doc_id": "qa_005",
            "doc_type": "qa",
            "question": "home loan ki qist kitni hogi 50 lakh pe",
            "answer": "Depends on interest rate and tenure. At 12% for 20 years, roughly 55,000 PKR monthly.",
            "category": "loans_credit",
            "language": "roman_urdu",
        },
    ]


@pytest.fixture
def sample_pdf_documents():
    """Sample PDF chunk documents for testing"""
    return [
        {
            "doc_id": "pdf_a1b2c3d4",
            "doc_type": "pdf_chunk",
            "retrieval_text": "Zakat Deduction Rules\n\nZakat is obligatory on all Muslims who own wealth above nisab threshold for one lunar year.",
            "chunk_text": "Zakat is obligatory on all Muslims who own wealth above nisab threshold for one lunar year.",
            "source_file": "sbp_islamic_banking_guide.pdf",
            "page_no": 42,
            "chunk_id": 1,
            "parent_section": "Zakat Deduction Rules",
            "category": "islamic_finance",
        },
        {
            "doc_id": "pdf_e5f6g7h8",
            "doc_type": "pdf_chunk",
            "retrieval_text": "Digital Payments\n\nEasyPaisa allows instant money transfer using mobile phone number.",
            "chunk_text": "EasyPaisa allows instant money transfer using mobile phone number.",
            "source_file": "digital_payments_act.pdf",
            "page_no": 15,
            "chunk_id": 3,
            "parent_section": "Mobile Wallet Services",
            "category": "digital_finance",
        },
    ]


@pytest.fixture
def mock_embeddings():
    """Generate deterministic mock embeddings for testing"""
    np.random.seed(42)
    return {
        "qa_001": np.random.randn(768).astype(np.float32),
        "qa_002": np.random.randn(768).astype(np.float32),
        "qa_003": np.random.randn(768).astype(np.float32),
        "qa_004": np.random.randn(768).astype(np.float32),
        "qa_005": np.random.randn(768).astype(np.float32),
        "pdf_a1b2c3d4": np.random.randn(768).astype(np.float32),
        "pdf_e5f6g7h8": np.random.randn(768).astype(np.float32),
        # Query embeddings
        "query_zakat": np.random.randn(768).astype(np.float32),
        "query_easypaisa": np.random.randn(768).astype(np.float32),
        "query_loan": np.random.randn(768).astype(np.float32),
    }


# =============================================================================
# Fixtures - Mock LLM Client
# =============================================================================

@pytest.fixture
def mock_groq_response():
    """Mock successful Groq API response"""
    return {
        "text": "Zakat is calculated as 2.5% of your total savings and assets that exceed the nisab threshold.",
        "prompt_tokens": 150,
        "completion_tokens": 25,
    }


@pytest.fixture
def mock_openai_response():
    """Mock successful OpenAI API response"""
    return {
        "text": "According to Islamic principles, zakat is obligatory on wealth held for one lunar year.",
        "prompt_tokens": 160,
        "completion_tokens": 30,
    }


@pytest.fixture
def mock_llm_client(mock_groq_response):
    """Create a fully mocked LLM client"""
    from generation.llm_client import LLMClient, LLMResponse
    
    def mock_generate(prompt):
        return LLMResponse(
            text=mock_groq_response["text"],
            provider="groq",
            model="llama-3.3-70b-versatile",
            prompt_tokens=mock_groq_response["prompt_tokens"],
            completion_tokens=mock_groq_response["completion_tokens"],
            latency_ms=150.0,
            retries_used=0,
            success=True,
        )
    
    client = MagicMock(spec=LLMClient)
    client.generate = mock_generate
    return client


# =============================================================================
# Fixtures - Temporary Directories
# =============================================================================

@pytest.fixture
def temp_artifacts_dir(tmp_path):
    """Create temporary artifacts directory for testing"""
    artifacts_dir = tmp_path / "test_artifacts"
    artifacts_dir.mkdir()
    return str(artifacts_dir)


@pytest.fixture
def temp_corpus_file(tmp_path, sample_qa_documents):
    """Create temporary corpus JSONL file"""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "test_corpus.jsonl"
    
    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc in sample_qa_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    return str(corpus_file)


# =============================================================================
# Fixtures - Cache
# =============================================================================

@pytest.fixture
def temp_cache():
    """Create a temporary semantic cache instance"""
    from cache.semantic_cache import SemanticCache
    
    cache = SemanticCache(
        l1_threshold=0.95,
        l2_threshold=0.90,
        ttl_seconds=3600,  # 1 hour for testing
        max_entries=100,
        corpus_version="test_v1",
    )
    return cache


# =============================================================================
# Fixtures - Environment
# =============================================================================

@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables before each test"""
    # Store original values
    orig_groq = os.environ.get("GROQ_API_KEY")
    orig_openai = os.environ.get("OPENAI_API_KEY")
    
    # Set test values
    os.environ["GROQ_API_KEY"] = "test_groq_key_gsk_xxx"
    os.environ["OPENAI_API_KEY"] = "test_openai_key_sk_xxx"
    
    yield
    
    # Restore original values
    if orig_groq is not None:
        os.environ["GROQ_API_KEY"] = orig_groq
    else:
        os.environ.pop("GROQ_API_KEY", None)
        
    if orig_openai is not None:
        os.environ["OPENAI_API_KEY"] = orig_openai
    else:
        os.environ.pop("OPENAI_API_KEY", None)


# =============================================================================
# Fixtures - Mock External APIs
# =============================================================================

@pytest.fixture
def mock_groq_api(mock_groq_response):
    """Mock the Groq API calls"""
    with patch("generation.llm_client._call_groq") as mock:
        mock.return_value = mock_groq_response
        yield mock


@pytest.fixture
def mock_openai_api(mock_openai_response):
    """Mock the OpenAI API calls"""
    with patch("generation.llm_client._call_openai") as mock:
        mock.return_value = mock_openai_response
        yield mock


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer encoding"""
    with patch("sentence_transformers.SentenceTransformer") as mock_class:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.random.randn(768).astype(np.float32)
        mock_class.return_value = mock_instance
        yield mock_class


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder for reranking"""
    with patch("sentence_transformers.CrossEncoder") as mock_class:
        mock_instance = MagicMock()
        # Return scores for query-doc pairs
        mock_instance.predict.return_value = np.array([0.85, 0.72, 0.91, 0.68, 0.55])
        mock_class.return_value = mock_instance
        yield mock_class


# =============================================================================
# Fixtures - Retrieval Pipeline Components
# =============================================================================

@pytest.fixture
def mock_vector_retriever(sample_qa_documents, mock_embeddings):
    """Create a mocked vector retriever"""
    from retrieval.vector_retriever import VectorRetriever
    
    retriever = MagicMock(spec=VectorRetriever)
    
    def mock_search(query_embedding, top_k=10, category_filter=None):
        # Return first top_k documents as mock results
        results = []
        for doc in sample_qa_documents[:top_k]:
            results.append({
                "doc_id": doc["doc_id"],
                "doc": doc,
                "score": 0.85,
            })
        return results
    
    retriever.search = mock_search
    retriever.encode_query = lambda q: mock_embeddings["query_zakat"]
    return retriever


@pytest.fixture
def mock_bm25_retriever(sample_qa_documents):
    """Create a mocked BM25 retriever"""
    from retrieval.bm25_retriever import BM25Retriever
    
    retriever = MagicMock(spec=BM25Retriever)
    
    def mock_search(query, top_k=10):
        # Return documents that might match
        results = []
        for doc in sample_qa_documents[:top_k]:
            results.append({
                "doc_id": doc["doc_id"],
                "doc": doc,
                "score": 0.75,
            })
        return results
    
    retriever.search = mock_search
    return retriever


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection - add markers based on test names"""
    for item in items:
        # Auto-mark slow tests
        if "adversarial" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        # Auto-mark integration tests
        if "integration" in item.nodeid or "e2e" in item.nodeid:
            item.add_marker(pytest.mark.integration)
