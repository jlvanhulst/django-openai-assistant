from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def mock_openai_client():
    with patch("openai.OpenAI") as mock_client:
        client = MagicMock()
        mock_client.return_value = client
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
