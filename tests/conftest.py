import os
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def mock_openai_client():
    with patch("openai.OpenAI") as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        client.api_key = os.environ.get("OPENAI_API_KEY", "mock-key")
        
        # Mock assistants list
        mock_assistant = {
            "id": "test_assistant_id",
            "name": "Test Assistant",
            "model": "gpt-4",
            "description": None,
            "instructions": None,
            "tools": [],
            "file_ids": [],
            "metadata": {},
            "created_at": 1234567890,
            "object": "assistant"
        }
        mock_list = MagicMock()
        mock_list.data = [mock_assistant]
        client.beta.assistants.list.return_value = mock_list
        client.beta.assistants.create.return_value = mock_assistant

        # Mock thread creation and retrieval
        mock_thread = {
            "id": "test_thread_id",
            "created_at": 1234567890,
            "metadata": {},
            "object": "thread"
        }
        client.beta.threads.create.return_value = mock_thread

        # Mock message creation and retrieval
        mock_message = {
            "id": "test_message_id",
            "thread_id": "test_thread_id",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": {
                    "value": "Test response",
                    "annotations": []
                }
            }],
            "file_ids": [],
            "assistant_id": "test_assistant_id",
            "run_id": "test_run_id",
            "created_at": 1234567890,
            "metadata": {},
            "object": "thread.message"
        }
        mock_messages = MagicMock()
        mock_messages.data = [mock_message]
        client.beta.threads.messages.list.return_value = mock_messages

        # Mock run creation and retrieval
        mock_run = {
            "id": "test_run_id",
            "thread_id": "test_thread_id",
            "assistant_id": "test_assistant_id",
            "status": "completed",
            "required_action": None,
            "last_error": None,
            "created_at": 1234567890,
            "expires_at": None,
            "started_at": 1234567890,
            "cancelled_at": None,
            "failed_at": None,
            "completed_at": 1234567890,
            "model": "gpt-4",
            "instructions": None,
            "tools": [],
            "file_ids": [],
            "metadata": {},
            "object": "thread.run"
        }
        client.beta.threads.runs.create.return_value = mock_run
        client.beta.threads.runs.retrieve.return_value = mock_run
        
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
