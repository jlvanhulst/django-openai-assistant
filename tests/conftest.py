from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def mock_openai_client():
    with patch("openai.OpenAI") as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        client.api_key = "test-api-key"
        client.beta.assistants.list.return_value.data = [
            MagicMock(name="Test Assistant", id="test_assistant_id")
        ]
        client.beta.threads.messages.list.return_value.data = [
            MagicMock(
                id="test_message_id",
                created_at=1234567890,
                thread_id="test_thread_id",
                role="assistant",
                content=[{"type": "text", "text": {"value": "Test response", "annotations": []}}],
                file_ids=[],
                assistant_id="test_assistant_id",
                run_id="test_run_id",
                metadata={},
                status="completed",
                object="thread.message"
            )
        ]
        yield client


@pytest.fixture
def mock_celery_task():
    with patch("celery.shared_task") as mock_task:
        yield mock_task


@pytest.fixture
def mock_redis():
    with patch("redis.Redis") as mock_redis:
        client = MagicMock()
        mock_redis.return_value = client
        yield client


@pytest.fixture
def test_settings():
    with patch("django.conf.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = "test-key"
        yield mock_settings
