import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from agent_core.llm.nvidia import NvidiaProvider, NvidiaTimeoutError, NvidiaBackendError
from agent_core.models import ChatMessage
from agent_core.config import Settings

@pytest.fixture
def settings(tmp_path):
    s = Settings(
        nvidia_api_key="test-key",
        nvidia_model="primary-model",
        nvidia_fallback_model="fallback-model",
        session_dir=tmp_path / "sessions",
        logs_dir=tmp_path / "logs",
        timeout_seconds=1.0  # Short timeout for testing
    )
    s.logs_dir.mkdir(parents=True, exist_ok=True)
    return s

@pytest.mark.asyncio
async def test_nvidia_fallback_on_timeout(settings):
    provider = NvidiaProvider(settings)
    provider.max_attempts = 1  # Speed up test
    
    # First call fails with timeout, second call (fallback) succeeds
    mock_responses = [
        httpx.TimeoutException("primary timeout"),
        MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "fallback success"}}]})
    ]
    
    with patch("httpx.AsyncClient.post", side_effect=mock_responses):
        response = await provider.chat([ChatMessage(role="user", content="hi")])
        assert response.content == "fallback success"
        # Since max_attempts=1, it should have tried primary once, then fallback once.
        # Total calls = 2

@pytest.mark.asyncio
async def test_nvidia_fallback_on_503(settings):
    provider = NvidiaProvider(settings)
    provider.max_attempts = 1
    
    # Mock for primary failing with 503
    mock_503 = MagicMock()
    mock_503.status_code = 503
    mock_503.raise_for_status.side_effect = httpx.HTTPStatusError("Service Unavailable", request=MagicMock(), response=mock_503)
    
    # Mock for fallback success
    mock_ok = MagicMock()
    mock_ok.status_code = 200
    mock_ok.json.return_value = {"choices": [{"message": {"content": "fallback success 503"}}]}
    
    with patch("httpx.AsyncClient.post", side_effect=[mock_503, mock_ok]):
        response = await provider.chat([ChatMessage(role="user", content="hi")])
        assert response.content == "fallback success 503"

@pytest.mark.asyncio
async def test_nvidia_total_failure(settings):
    provider = NvidiaProvider(settings)
    provider.max_attempts = 1
    
    # Both fail
    mock_fail = httpx.ConnectError("connection fail")
    
    with patch("httpx.AsyncClient.post", side_effect=[mock_fail, mock_fail]):
        with pytest.raises(NvidiaBackendError):
            await provider.chat([ChatMessage(role="user", content="hi")])
