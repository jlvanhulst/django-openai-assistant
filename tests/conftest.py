from unittest.mock import MagicMock, patch

import pytest
from django.conf import settings  # noqa: F401
from openai import OpenAI  # noqa: F401


@pytest.fixture
def mock_celery_task():
    with patch("django_openai_assistant.assistant.get_status") as mock_task:
        mock_task.delay = MagicMock()
        yield mock_task


@pytest.fixture(autouse=True)
def mock_redis():
    with patch("redis.Redis") as mock_redis:
        mock_redis.return_value.ping.return_value = True
        mock_redis.return_value = MagicMock()
        yield mock_redis


@pytest.fixture
def mock_openai_client():
    with patch("django_openai_assistant.assistant.OpenAI") as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        client.api_key = "test-key"
        client.beta.assistants.list.return_value = MagicMock(data=[])
        client.beta.threads.runs.create.return_value = MagicMock(id="run_123")
        yield client


@pytest.fixture
def test_settings():
    with patch("django.conf.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = "test-key"
        yield mock_settings
