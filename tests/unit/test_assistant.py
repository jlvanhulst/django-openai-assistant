from unittest.mock import MagicMock

import pytest
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Message, Run
from pydantic import BaseModel

from django_openai_assistant.assistant import (
    asmarkdown,
    assistantTask,
    createAssistant,
    getAssistant,
    set_default_tools,
)
from django_openai_assistant.models import OpenaiTask


@pytest.fixture
def mock_assistant():
    return Assistant(
        id="asst_123",
        created_at=1234567890,
        name="Test Assistant",
        description=None,
        model="gpt-4",
        instructions=None,
        tools=[],
        file_ids=[],
        metadata={},
        object="assistant",
    )


@pytest.fixture
def mock_thread():
    return Thread(id="thread_123", created_at=1234567890, metadata={}, object="thread")


@pytest.fixture
def mock_run():
    return Run(
        id="run_123",
        created_at=1234567890,
        thread_id="thread_123",
        assistant_id="asst_123",
        status="completed",
        required_action=None,
        last_error=None,
        expires_at=None,
        started_at=1234567890,
        cancelled_at=None,
        failed_at=None,
        completed_at=1234567890,
        model="gpt-4",
        instructions="Test instructions",
        tools=[],
        file_ids=[],
        metadata={},
        object="thread.run",
        parallel_tool_calls=0,
    )


def test_assistant_creation_and_configuration(mock_openai_client, mock_assistant):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.assistants.create.return_value = mock_assistant

    # Test basic initialization
    task = assistantTask(assistantName="Test Assistant")
    assert task.assistant_id == "asst_123"
    assert task.tools == []  # tools is empty list by default
    assert task.metadata == {}

    # Test with completion callback
    task = assistantTask(
        assistantName="Test Assistant",
        completionCall="test_callback",
        metadata={"user": "test@example.com"},
    )
    assert task.completionCall == "test_callback"
    assert task.metadata == {"user": "test@example.com"}

    # Test assistant creation with tools
    tools = {
        "fuzzy_search": {"module": "pinecone"},
        "create_event": {"module": "calendar"},
    }
    new_assistant = createAssistant(
        name="New Assistant",
        tools=tools,
        model="gpt-4",
        instructions="Test instructions",
    )
    assert new_assistant.id == "asst_123"
    assert new_assistant.model == "gpt-4"

    # Test getting existing assistants
    assistants = getAssistant()
    assert len(assistants) > 0
    assert assistants[0].id == "asst_123"


def test_assistant_task_with_tools(mock_openai_client, mock_assistant):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    tools = ["test_module:test_function"]

    task = assistantTask(assistantName="Test Assistant", tools=tools)
    assert task.tools == tools
    assert isinstance(task.metadata, dict)


def test_set_default_tools():
    # Test with dictionary format (vicbot style)
    tools = {
        "getCompany": {"module": "salesforce"},
        "fuzzy_search": {"module": "pinecone"},
        "create_event": {"module": "calendar"},
    }
    result = set_default_tools(tools=tools, package="chatbot")
    assert "getCompany" in result
    assert result["getCompany"]["module"] == "salesforce"
    assert "fuzzy_search" in result
    assert "create_event" in result

    # Test with legacy format
    tools_list = ["test_module:test_function"]
    result = set_default_tools(tools=tools_list)
    assert "test_function" in result
    assert result["test_function"]["module"] == "test_module"


def test_create_run(mock_openai_client, mock_assistant, mock_thread, mock_run):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    mock_openai_client.beta.threads.runs.create.return_value = mock_run

    task = assistantTask(assistantName="Test Assistant")
    task.prompt = "Test prompt"
    task.threadObject = mock_thread
    task.task = MagicMock(threadId="thread_123")
    run_id = task.createRun(temperature=0.7)

    assert run_id == "run_123"
    assert task.thread_id == "thread_123"
    assert task.run_id == "run_123"


@pytest.mark.django_db
def test_tool_calling_and_completion(
    mock_openai_client, mock_assistant, mock_thread, mock_run
):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test tool calling with Pydantic model
    class TestParams(BaseModel):
        company_name: str
        email: str = None

    tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "getCompany",
            "arguments": '{"company_name": "Test Corp", "email": "test@example.com"}',
        },
    }
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[tool_call])
    )
    mock_openai_client.beta.threads.runs.create.return_value = mock_run

    # Test with real-world tool configuration
    tools = ["salesforce:getCompany", "pinecone:fuzzy_search"]
    task = assistantTask(
        assistantName="Test Assistant",
        tools=tools,
        metadata={"user_email": "test@example.com"},
    )
    task.prompt = "Get company information"
    task.threadObject = mock_thread
    task.task = MagicMock(threadId="thread_123")
    run_id = task.createRun()

    assert run_id == "run_123"
    assert task.status == "requires_action"
    assert task.metadata["user_email"] == "test@example.com"

    # Test completion callback
    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run

    task = assistantTask(assistantName="Test Assistant", completionCall="test_callback")
    task.run_id = run_id
    task.thread_id = "thread_123"

    status = task.get_run_status()
    assert status == "completed"
    assert task.completionCall == "test_callback"


def test_thread_message_handling(mock_openai_client, mock_assistant, mock_thread):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test message with text and file content
    mock_message = Message(
        id="msg_123",
        created_at=1234567890,
        thread_id="thread_123",
        role="assistant",
        content=[
            {"type": "text", "text": {"value": "Text response", "annotations": []}},
            {"type": "image_file", "image_file": {"file_id": "file_123"}},
        ],
        file_ids=["file_123"],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={},
        object="thread.message",
        status="completed",
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]

    task = assistantTask(assistantName="Test Assistant")
    task.threadObject = mock_thread
    task.task = MagicMock(threadId="thread_123")

    # Test message retrieval
    messages = task.getAllMessages()
    assert len(messages) == 1
    assert messages[0].id == "msg_123"
    assert len(messages[0].content) == 2

    # Test full response formatting
    full_response = task.getFullResponse()
    assert "Text response" in full_response

    # Test file retrieval
    mock_openai_client.files.content.return_value = b"file content"
    file_content = task.retrieveFile("file_123")
    assert file_content == b"file content"


def test_run_status_tracking(mock_openai_client, mock_assistant, mock_thread, mock_run):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test various run statuses
    statuses = ["queued", "in_progress", "requires_action", "completed", "failed"]
    for status in statuses:
        mock_run.status = status
        mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run

        task = assistantTask(
            assistantName="Test Assistant",
            completionCall="test_callback" if status == "completed" else None,
        )
        task.runObject = mock_run
        task.threadObject = mock_thread
        task.task = MagicMock(runId="run_123", threadId="thread_123")

        current_status = task.getRunStatus()
        assert current_status == status

        if status == "completed":
            assert task.completionCall == "test_callback"
        elif status == "requires_action":
            assert task.status == "requires_action"


def test_file_upload(mock_openai_client, mock_assistant):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.files.create.return_value = MagicMock(id="file_123")

    task = assistantTask(assistantName="Test Assistant")
    fileContent = b"test content"
    file_id = task.uploadFile(fileContent=fileContent, filename="test.txt")

    assert file_id == "file_123"
    assert len(task._fileids) == 1
    assert task._fileids[0]["vision"] is False
    assert task._fileids[0]["retrieval"] is True


def test_vision_file_upload(mock_openai_client, mock_assistant):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.files.create.return_value = MagicMock(id="file_123")

    task = assistantTask(assistantName="Test Assistant")
    fileContent = b"test image"
    file_id = task.uploadFile(fileContent=fileContent, filename="test.jpg")

    assert file_id == "file_123"
    assert len(task._fileids) == 1
    assert task._fileids[0]["vision"] is True


def test_message_handling(mock_openai_client, mock_assistant, mock_thread):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    mock_message = Message(
        id="msg_123",
        created_at=1234567890,
        thread_id="thread_123",
        role="assistant",
        content=[
            {"type": "text", "text": {"value": "Test response", "annotations": []}}
        ],
        file_ids=[],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={},
        object="thread.message",
        status="completed",
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]

    task = assistantTask(assistantName="Test Assistant")
    task.threadObject = mock_thread
    task.task = MagicMock(threadId="thread_123")

    # Test message retrieval methods
    last_response = task.getLastResponse()
    assert last_response is not None
    assert last_response.id == "msg_123"

    all_messages = task.getAllMessages()
    assert len(all_messages) == 1
    assert all_messages[0].id == "msg_123"

    full_response = task.getFullResponse()
    assert full_response == "Test response"


def test_response_formats(mock_openai_client, mock_assistant, mock_thread):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant")

    # Test JSON response parsing
    task.response = '{"key": "value"}'
    json_response = task.jsonResponse()
    assert json_response == {"key": "value"}

    # Test markdown code block parsing
    task.response = '```json\n{"key": "value"}\n```'
    json_response = task.jsonResponse()
    assert json_response == {"key": "value"}

    # Test markdown with replacements
    task.response = "**bold** *italic*"
    markdown_response = task.getMarkdownResponse(replaceThis="bold", withThis="strong")
    assert "<p><strong>strong</strong> <em>italic</em></p>" == markdown_response

    # Test null responses
    task.response = None
    assert task.jsonResponse() is None
    assert task.getMarkdownResponse() is None


def test_asmarkdown_function():
    test_string = "**bold** *italic*"
    result = asmarkdown(test_string)
    assert "bold" in result
    assert "italic" in result

    result_with_replace = asmarkdown(test_string, replaceThis="bold", withThis="strong")
    assert "strong" in result_with_replace
    assert "italic" in result_with_replace


@pytest.mark.django_db
def test_celery_integration(
    mock_openai_client, mock_assistant, mock_thread, mock_run, mock_celery_task
):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test async task queuing
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    task = assistantTask(
        assistantName="Test Assistant",
        metadata={"user_email": "test@example.com"},
        completionCall="chatbot.google:replyToEmail",
    )
    task.prompt = "Process this email"
    task.threadObject = mock_thread
    task.task = MagicMock(threadId="thread_123")
    run_id = task.createRun()

    mock_celery_task.delay.assert_called_once_with(
        run_id=run_id,
        thread_id=task.task.threadId,
        completionCall="chatbot.google:replyToEmail",
    )

    # Test task status monitoring
    mock_run.status = "in_progress"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId=run_id,
        threadId="thread_123",
        assistant_id="asst_123",
        status="in_progress",
    )
    task.threadObject = mock_thread
    task.runObject = mock_run

    from django_openai_assistant.assistant import get_status

    assert get_status(f"{task.run_id},{task.thread_id}") == "in_progress"
    assert task.task.status == "in_progress"

    # Test completion callback
    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task.status = "completed"
    task.task.completionCall = "chatbot.google:replyToEmail"
    task.task.save()

    mock_celery_task.delay.assert_called_with(
        run_id=run_id,
        thread_id=task.thread_id,
        completionCall="chatbot.google:replyToEmail",
    )
    assert get_status(f"{task.run_id},{task.thread_id}") == "completed"

    # Test error handling in task
    mock_run.status = "failed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId=run_id,
        threadId="thread_123",
        assistant_id="asst_123",
        status="failed",
    )

    assert get_status(f"{task.run_id},{task.thread_id}") == "failed"
    assert task.task.status == "failed"


@pytest.mark.django_db
def test_vision_support(mock_openai_client, mock_assistant, mock_thread, mock_run):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant")

    # Test image file upload with vision support
    mock_image = MagicMock()
    mock_image.name = "test.png"
    mock_image.read.return_value = b"image content"

    mock_openai_client.files.create.return_value = MagicMock(
        id="file_123",
        filename="test.png",
        bytes=len(b"image content"),
        purpose="assistants",
        created_at=1234567890,
        content_type="image/png",
    )

    file_id = task.uploadFile(file=mock_image, filename="test.png")
    assert file_id == "file_123"
    assert task._fileids[-1]["vision"] is True
    assert task._fileids[-1]["retrieval"] is False

    # Test vision analysis with image
    mock_message = Message(
        id="msg_123",
        created_at=1234567890,
        thread_id="thread_123",
        role="assistant",
        content=[
            {"type": "image_file", "image_file": {"file_id": "file_123"}},
            {
                "type": "text",
                "text": {"value": "The image shows a test pattern", "annotations": []},
            },
        ],
        file_ids=["file_123"],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={},
        object="thread.message",
        status="completed",
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]

    task.prompt = "Analyze this image"
    task.threadObject = mock_thread
    task.task = MagicMock(threadId="thread_123")
    run_id = task.createRun()
    assert run_id == mock_run.id

    response = task.getFullResponse()
    assert "The image shows a test pattern" in response

    # Test multiple image handling
    mock_image2 = MagicMock()
    mock_image2.name = "test2.jpg"
    mock_image2.read.return_value = b"second image content"

    mock_openai_client.files.create.return_value = MagicMock(
        id="file_456",
        filename="test2.jpg",
        bytes=len(b"second image content"),
        purpose="assistants",
        created_at=1234567890,
        content_type="image/jpeg",
    )

    file_id2 = task.uploadFile(file=mock_image2, filename="test2.jpg")
    assert file_id2 == "file_456"
    assert task._fileids[-1]["vision"] is True

    # Test email with inline images
    mock_message.content = [
        {
            "type": "text",
            "text": {"value": "Here are the analyzed images:", "annotations": []},
        },
        {"type": "image_file", "image_file": {"file_id": "file_123"}},
        {"type": "image_file", "image_file": {"file_id": "file_456"}},
    ]
    mock_message.file_ids = ["file_123", "file_456"]

    response = task.getFullResponse()
    assert "Here are the analyzed images" in response

    # Test image file deletion
    task.deleteFile("file_123")
    mock_openai_client.files.delete.assert_called_with(file_id="file_123")

    # Test vision support with different file types
    image_types = [".png", ".jpg", ".jpeg", ".webp"]
    for ext in image_types:
        mock_file = MagicMock()
        mock_file.name = f"test{ext}"
        mock_file.read.return_value = b"image content"
        file_id = task.uploadFile(file=mock_file, filename=f"test{ext}")
        assert task._fileids[-1]["vision"] is True
        assert task._fileids[-1]["retrieval"] is False


def test_error_handling(mock_openai_client, mock_assistant, mock_thread, mock_run):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test missing thread ID
    task = assistantTask(assistantName="Test Assistant")
    with pytest.raises(ValueError, match="Thread ID is required"):
        task.getAllMessages()

    # Test invalid assistant name
    mock_openai_client.beta.assistants.list.return_value.data = []
    with pytest.raises(ValueError, match="Assistant .* not found"):
        assistantTask(assistantName="NonexistentAssistant")

    # Test run creation failure
    mock_openai_client.beta.threads.runs.create.side_effect = Exception("API Error")
    task = assistantTask(assistantName="Test Assistant")
    task.threadObject = mock_thread
    task.task = MagicMock(threadId="thread_123")
    assert task.createRun() is None

    # Test invalid tool configuration
    mock_openai_client.beta.assistants.list.return_value.data = []
    with pytest.raises(ValueError, match="Assistant .* not found"):
        assistantTask(assistantName="Test Assistant", tools="invalid")

    # Test run status handling
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_run.status = "failed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task = assistantTask(assistantName="Test Assistant")
    task.runObject = mock_run
    task.task = MagicMock(runId="run_123")
    status = task.getRunStatus()
    assert status == "failed"
