"""
test_llm_client.py - Tests for LLM client with mocked API calls
"""

import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
@pytest.mark.llm
class TestLLMClient:
    """Test suite for LLM client with mocked API providers"""
    
    def test_llm_client_initialization(self, base_config):
        """Test LLM client initializes with correct configuration"""
        from generation.llm_client import LLMClient
        
        client = LLMClient(base_config)
        
        assert client.primary_provider == "groq"
        assert client.primary_model == "llama-3.3-70b-versatile"
        assert client.fallback_enabled is True
        assert client.fallback_provider == "openai"
        assert client.max_tokens == 300
        assert client.temperature == 0.1
    
    def test_successful_groq_call(self, base_config, mock_groq_api, mock_groq_response):
        """Test successful generation via Groq"""
        from generation.llm_client import LLMClient
        
        client = LLMClient(base_config)
        response = client.generate("What is zakat?")
        
        assert response.success is True
        assert response.provider == "groq"
        assert response.model == "llama-3.3-70b-versatile"
        assert response.text == mock_groq_response["text"]
        assert response.prompt_tokens == mock_groq_response["prompt_tokens"]
        assert response.completion_tokens == mock_groq_response["completion_tokens"]
        assert response.retries_used == 0
        assert response.error is None
        
        # Verify API was called
        mock_groq_api.assert_called_once()
    
    def test_successful_openai_fallback(self, base_config, mock_openai_api, mock_openai_response):
        """Test fallback to OpenAI when Groq fails"""
        from generation.llm_client import LLMClient
        
        # Make Groq fail
        mock_groq_api = MagicMock(side_effect=Exception("rate_limit_exceeded"))
        
        with patch("generation.llm_client._call_groq", mock_groq_api):
            client = LLMClient(base_config)
            response = client.generate("What is zakat?")
        
        assert response.success is True
        assert response.provider == "openai"
        assert response.model == "gpt-4o-mini"
        assert response.text == mock_openai_response["text"]
    
    def test_retry_on_rate_limit(self, base_config, mock_groq_response):
        """Test exponential backoff retry on rate limit errors"""
        from generation.llm_client import LLMClient
        
        call_count = 0
        
        def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("rate_limit_exceeded")
            return mock_groq_response
        
        with patch("generation.llm_client._call_groq", side_effect=fail_then_succeed):
            with patch("time.sleep") as mock_sleep:  # Don't actually sleep
                client = LLMClient(base_config)
                response = client.generate("What is zakat?")
        
        assert response.success is True
        assert call_count == 2  # Failed once, succeeded on retry
        assert response.retries_used >= 1
    
    def test_retry_on_timeout(self, base_config, mock_groq_response):
        """Test retry on timeout errors"""
        from generation.llm_client import LLMClient
        
        call_count = 0
        
        def timeout_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Request timed out")
            return mock_groq_response
        
        with patch("generation.llm_client._call_groq", side_effect=timeout_then_succeed):
            with patch("time.sleep"):  # Don't actually sleep
                client = LLMClient(base_config)
                response = client.generate("What is zakat?")
        
        assert response.success is True
        assert call_count == 2
    
    def test_no_retry_on_fatal_error(self, base_config):
        """Test no retry on non-retryable errors"""
        from generation.llm_client import LLMClient
        
        # Invalid API key should not retry
        with patch("generation.llm_client._call_groq", side_effect=Exception("invalid_api_key")):
            with patch("generation.llm_client._call_openai") as mock_openai:
                client = LLMClient(base_config)
                response = client.generate("What is zakat?")
        
        # Should fail without retry, then fallback to OpenAI
        mock_openai.assert_called_once()
    
    def test_both_providers_fail(self, base_config):
        """Test graceful failure when both providers fail"""
        from generation.llm_client import LLMClient
        
        with patch("generation.llm_client._call_groq", side_effect=Exception("server_error")):
            with patch("generation.llm_client._call_openai", side_effect=Exception("server_error")):
                with patch("time.sleep"):  # Don't sleep
                    client = LLMClient(base_config)
                    response = client.generate("What is zakat?")
        
        assert response.success is False
        assert response.error is not None
        assert "Both providers failed" in response.error
        assert response.text == ""
    
    def test_hard_timeout(self, base_config):
        """Test hard timeout enforcement"""
        from generation.llm_client import LLMClient
        
        def slow_call(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow API
            return {"text": "response", "prompt_tokens": 10, "completion_tokens": 5}
        
        with patch("generation.llm_client._call_groq", side_effect=slow_call):
            client = LLMClient(base_config)
            client.timeout = 0.05  # Very short timeout for testing
            
            response = client.generate("What is zakat?")
            
            # Should fail or fallback
            assert response.success is False or response.provider == "openai"
    
    def test_latency_tracking(self, base_config, mock_groq_api, mock_groq_response):
        """Test latency is properly tracked"""
        from generation.llm_client import LLMClient
        
        with patch("time.time") as mock_time:
            mock_time.side_effect = [0.0, 0.15]  # Start, end (150ms)
            
            client = LLMClient(base_config)
            response = client.generate("What is zakat?")
        
        assert response.latency_ms == 150.0
    
    def test_fallback_disabled(self, base_config):
        """Test behavior when fallback is disabled"""
        from generation.llm_client import LLMClient
        
        base_config["generation"]["fallback_enabled"] = False
        
        with patch("generation.llm_client._call_groq", side_effect=Exception("server_error")):
            with patch("time.sleep"):
                client = LLMClient(base_config)
                response = client.generate("What is zakat?")
        
        assert response.success is False
        assert "fallback disabled" in response.error
    
    def test_unknown_provider(self, base_config):
        """Test error on unknown provider"""
        from generation.llm_client import LLMClient
        
        base_config["generation"]["primary_provider"] = "unknown_provider"
        
        client = LLMClient(base_config)
        response = client.generate("What is zakat?")
        
        assert response.success is False
        assert "Unknown primary provider" in response.error


@pytest.mark.unit
@pytest.mark.llm
class TestLLMResponse:
    """Test LLMResponse dataclass"""
    
    def test_response_creation(self):
        """Test creating LLMResponse"""
        from generation.llm_client import LLMResponse
        
        response = LLMResponse(
            text="Test response",
            provider="groq",
            model="llama-3.3-70b",
            prompt_tokens=100,
            completion_tokens=20,
            latency_ms=200.0,
            retries_used=1,
            success=True,
        )
        
        assert response.text == "Test response"
        assert response.success is True
    
    def test_error_response(self):
        """Test error response creation"""
        from generation.llm_client import LLMResponse
        
        response = LLMResponse(
            text="",
            provider="none",
            model="none",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0.0,
            retries_used=0,
            success=False,
            error="API key invalid",
        )
        
        assert response.success is False
        assert response.error == "API key invalid"


@pytest.mark.unit
class TestRetryLogic:
    """Test retry and backoff logic"""
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation"""
        from generation.llm_client import _exponential_backoff
        
        assert _exponential_backoff(0) == 0.5
        assert _exponential_backoff(1) == 1.0
        assert _exponential_backoff(2) == 2.0
        assert _exponential_backoff(3) == 4.0
        assert _exponential_backoff(4) == 8.0
        # Cap at 8 seconds
        assert _exponential_backoff(10) == 8.0
    
    def test_retryable_exception_detection(self):
        """Test detection of retryable exceptions"""
        from generation.llm_client import _is_retryable
        
        assert _is_retryable(Exception("rate_limit exceeded")) is True
        assert _is_retryable(Exception("timeout occurred")) is True
        assert _is_retryable(Exception("connection error")) is True
        assert _is_retryable(Exception("server_error 500")) is True
        assert _is_retryable(Exception("service_unavailable")) is True
        assert _is_retryable(TimeoutError()) is True
        assert _is_retryable(ConnectionError()) is True
        
        assert _is_retryable(Exception("invalid api key")) is False
        assert _is_retryable(Exception("bad request")) is False


@pytest.mark.integration
class TestLLMClientIntegration:
    """Integration tests that may call actual APIs in CI with real keys"""
    
    @pytest.mark.skip(
        reason="Integration tests disabled. Use --run-integration to enable."
    )
    def test_real_groq_call(self, base_config):
        """Test with real Groq API (requires GROQ_API_KEY)"""
        import os
        
        if not os.environ.get("GROQ_API_KEY", "").startswith("gsk_"):
            pytest.skip("Real GROQ_API_KEY not set")
        
        from generation.llm_client import LLMClient
        
        client = LLMClient(base_config)
        response = client.generate("What is 2+2? Answer in one word.")
        
        assert response.success is True
        assert "4" in response.text or "four" in response.text.lower()
