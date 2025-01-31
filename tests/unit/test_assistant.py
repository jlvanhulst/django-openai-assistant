import json
from unittest.mock import MagicMock

import openai
import pytest
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Message, Run
from openai.types.beta.threads.runs import ToolCall

from django_openai_assistant.assistant import (
    asmarkdown,
    assistantTask,
    _create_assistant,
    get_assistant,
    set_default_tools,
    _call_tools_delay,
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
        instructions="",  # Empty string instead of None
        tools=[],
        file_ids=[],
        metadata={},
        object="thread.run",
        parallel_tool_calls=0  # Required field
    )


def test_assistant_creation_and_configuration(
    mock_openai_client, mock_assistant, mock_thread, mock_run
):
    """Test core assistant initialization and configuration.

    This test verifies:
    1. Basic initialization with different parameters
    2. Metadata and completion call handling
    3. Assistant creation with tools

    Does NOT test:
    - Tool implementations
    - External service integration
    - Complex configuration scenarios"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]

    # Test initialization with comboId
    combo_id = f"{mock_run.id},{mock_thread.id}"
    task = assistantTask(comboId=combo_id)
    assert task.run_id == mock_run.id
    assert task.thread_id == mock_thread.id

    # Test initialization with metadata
    metadata = {"id": "test123", "sobject": "Account"}
    task = assistantTask(
        assistantName="Test Assistant",
        metadata=metadata,
        completionCall="test:callback",
    )
    assert task.metadata == metadata
    assert task.completion_call == "test:callback"
    mock_openai_client.beta.assistants.create.return_value = mock_assistant

    # Test basic initialization
    task = assistantTask(assistantName="Test Assistant")
    assert task.assistant_id == "asst_123"
    assert task.tools == []
    assert task.metadata == {}

    # Test with completion callback
    task = assistantTask(
        assistantName="Test Assistant",
        completionCall="test_callback",
        metadata={"user": "test@example.com"},
    )
    assert task.completion_call == "test_callback"
    assert task.metadata == {"user": "test@example.com"}

    # Test assistant creation with tools
    tools = {
        "generic_tool1": {"module": "module1"},
        "generic_tool2": {"module": "module2"},
    }
    new_assistant = _create_assistant(
        name="New Assistant",
        tools=list(tools.keys()),
        model="gpt-4",
        instructions="Test assistant instructions",
    )
    assert new_assistant.id == "asst_123"
    assert new_assistant.model == "gpt-4"
    assert len(new_assistant.tools) == len(tools)

    # Test getting existing assistants
    assistants = get_assistant()
    assert len(assistants) > 0
    assert assistants[0].id == "asst_123"
    assert isinstance(assistants[0].tools, list)


def test_assistant_task_with_tools(mock_openai_client, mock_assistant):
    """Test the assistantTask's ability to configure and store tool settings.

    This test verifies:
    1. Tool list storage and validation
    2. Empty tool list handling
    3. Single tool configuration

    Does NOT test actual tool implementations or external module functionality."""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    tools = ["module:function1", "module:function2"]  # Use proper tool format

    # Verify tool configuration storage
    task = assistantTask(assistantName="Test Assistant", tools=tools)
    assert task.tools == tools
    assert isinstance(task.metadata, dict)

    # Verify empty tool handling
    task_no_tools = assistantTask(assistantName="Test Assistant")
    assert task_no_tools.tools == []

    # Verify single tool handling
    task_single_tool = assistantTask(
        assistantName="Test Assistant", tools=["function1"]
    )
    assert isinstance(task_single_tool.tools, list)
    assert len(task_single_tool.tools) == 1
    assert task_single_tool.tools[0] == "function1"


def test_set_default_tools():
    """Test the core tool configuration parsing functionality.

    This test verifies:
    1. Dictionary format tool configuration parsing
    2. Legacy format tool configuration parsing
    3. Tool module mapping

    Does NOT test actual tool loading or execution."""

    # Test dictionary format parsing
    tools = {
        "function1": {"module": "core"},
        "function2": {"module": "core"},
    }
    result = set_default_tools(tools=list(tools.keys()), package="test")
    assert "function1" in result
    assert result["function1"]["module"] == "core"
    assert "function2" in result
    assert result["function2"]["module"] == "core"

    # Test legacy format parsing
    tools_list = ["core:function"]
    result = set_default_tools(tools=tools_list)
    assert "function" in result
    assert result["function"]["module"] == "core"


def test_create_run(mock_openai_client, mock_assistant, mock_thread, mock_run):
    """Test core run creation functionality.

    This test verifies:
    1. Basic run creation with parameters
    2. Thread and run ID assignment
    3. Task object creation

    Does NOT test:
    - Run execution
    - External API interactions
    - Complex run configurations"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    mock_openai_client.beta.threads.runs.create.return_value = mock_run

    task = assistantTask(assistantName="Test Assistant")
    task.prompt = "Test prompt"
    run_id = task.create_run(temperature=0.7)

    assert run_id == "run_123"
    assert task.threadObject and task.threadObject.id == "thread_123"
    assert task.task.runId == "run_123"


def test_tool_call_handling(
    mock_openai_client, mock_assistant, mock_thread, mock_run, mock_celery_task
):
    """Test assistantTask's core tool call handling.
    This test verifies:
    1. Basic tool call detection
    2. Tool call scheduling via call_tools_delay
    3. Run status updates

    Does NOT test:
    - Tool implementations
    - External services
    - Business logic"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test basic tool call handling
    tool_call = ToolCall(
        id="call_123",
        type="function",
        function=MagicMock(
            name="test_tool",
            arguments=json.dumps({"param": "value"}),
        ),
    )
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[tool_call])
    )
    mock_openai_client.beta.threads.runs.create.return_value = mock_run

    task = assistantTask(
        assistantName="Test Assistant",
        tools=["test_tool"],
    )
    task.prompt = "Test prompt"
    run_id = task.create_run()
    assert run_id == "run_123"
    assert task.status == "requires_action"

    # Verify tool scheduling
    combo_id = f"{task.run_id},{task.thread_id}"
    _call_tools_delay(combo_id, tool_calls=[tool_call], tools=task.tools)
    mock_celery_task.delay.assert_called_once()

    # Verify status update after completion
    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId=run_id,
        threadId=task.thread_id,
        assistant_id=task.assistant_id,
        status="completed",
    )
    assert task.task.status == "completed"


def test_thread_message_handling(mock_openai_client, mock_assistant, mock_thread):
    """Test core message handling functionality.

    This test verifies:
    1. Basic message retrieval
    2. Message content extraction
    3. File ID handling in messages

    Does NOT test:
    - Message processing logic
    - External API interactions
    - Complex message scenarios"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test message with text and file content
    mock_message = Message(
        id="msg_123",
        created_at=1234567890,
        thread_id="thread_123",
        role="assistant",
        content=[
            {"type": "text", "text": {"value": "Text response"}},
            {"type": "image_file", "image_file": {"file_id": "file_123"}}
        ],
        file_ids=["file_123"],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={},
        status="completed",  # Add required status field
        object="thread.message"
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]

    task = assistantTask(assistantName="Test Assistant")
    task.threadObject = mock_thread

    # Test message retrieval
    messages = task.get_all_messages()
    assert len(messages) == 1
    message = messages[0]
    assert isinstance(message, dict)
    assert message["id"] == "msg_123"
    assert len(message["content"]) == 2

    # Test full response formatting
    full_response = task.get_full_response()
    assert "Text response" in full_response

    # Test file retrieval
    mock_openai_client.files.content.return_value = b"file content"
    file_content = task.retrieve_file("file_123")
    assert file_content == b"file content"


def test_metadata_management(mock_openai_client, mock_assistant, mock_thread, mock_run):
    """Test core metadata functionality of assistantTask.

    This test verifies:
    1. Basic metadata storage
    2. Metadata retrieval
    3. Metadata isolation between instances

    Does NOT test:
    - External metadata handling
    - Complex metadata structures
    - Metadata validation"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test basic metadata storage
    test_metadata = {"key": "value"}
    task = assistantTask(
        assistantName="Test Assistant",
        metadata=test_metadata,
    )
    assert task.metadata == test_metadata

    # Test metadata persistence
    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId="run_123",
        threadId="thread_123",
        assistant_id="asst_123",
        metadata=test_metadata,
        status="completed",
    )
    assert task.metadata == test_metadata

    # Test metadata isolation
    task2 = assistantTask(
        assistantName="Test Assistant",
        metadata={"other": "value"},
    )
    assert task2.metadata != task.metadata
    assert task2.metadata == {"other": "value"}


def test_file_upload(mock_openai_client, mock_assistant):
    """Test basic file upload functionality.

    This test verifies:
    1. File upload with content
    2. File upload with file object
    3. File ID tracking

    Does NOT test:
    - File processing
    - External services
    - Complex file operations"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.files.create.return_value = MagicMock(
        id="file_123", purpose="assistants"
    )

    task = assistantTask(assistantName="Test Assistant")

    # Test content upload
    content = b"test content"
    file_id = task.upload_file(file_content=content, filename="test.txt")
    assert file_id == "file_123"
    assert len(task._fileids) == 1

    # Test file object upload
    mock_file = MagicMock()
    mock_file.name = "test.txt"
    mock_file.read.return_value = b"test content"
    file_id = task.upload_file(file=mock_file)
    assert file_id == "file_123"
    assert len(task._fileids) == 2

    # Verify OpenAI API interaction
    mock_openai_client.files.create.assert_called_with(
        file=mock_file.read.return_value, purpose="assistants", filename="test.txt"
    )


def test_message_handling(mock_openai_client, mock_assistant, mock_thread):
    """Test core message retrieval methods.

    This test verifies:
    1. Last response retrieval
    2. All messages retrieval
    3. Full response concatenation

    Does NOT test:
    - Message content validation
    - Message ordering logic
    - External API response handling"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    mock_message = Message(
        id="msg_123",
        created_at=1234567890,
        thread_id="thread_123",
        role="assistant",
        content=[{"type": "text", "text": {"value": "Test response"}}],
        file_ids=[],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={},
        status="completed",
        object="thread.message"
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]

    task = assistantTask(assistantName="Test Assistant")
    task.threadObject = mock_thread

    # Test message retrieval methods
    last_response = task.get_last_response()
    assert last_response is not None
    assert isinstance(last_response, dict)
    assert last_response["id"] == "msg_123"

    all_messages = task.get_all_messages()
    assert len(all_messages) == 1
    message = all_messages[0]
    assert isinstance(message, dict)
    assert message["id"] == "msg_123"

    full_response = task.get_full_response()
    assert full_response == "Test response"


def test_response_formats(mock_openai_client, mock_assistant, mock_thread):
    """Test core response formatting functionality.

    This test verifies:
    1. Text response formatting
    2. JSON response parsing
    3. Markdown response handling
    4. Multi-part message handling

    Does NOT test:
    - Response content validation
    - External format processing
    - Complex formatting scenarios"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant")
    task.threadObject = mock_thread

    # Test streaming response
    mock_message = Message(
        id="msg_123",
        created_at=1234567890,
        thread_id="thread_123",
        role="assistant",
        content=[
            {"type": "text", "text": {"value": "Part 1"}},
            {"type": "text", "text": {"value": "Part 2"}},
            {"type": "text", "text": {"value": "Part 3"}}
        ],
        file_ids=[],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={},
        status="completed",
        object="thread.message"
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]

    # Test multi-message response
    full_response = task.get_full_response()
    assert "Part 1" in full_response
    assert "Part 2" in full_response
    assert "Part 3" in full_response

    # Test JSON response parsing
    task.response = '{"key": "value", "nested": {"array": [1, 2, 3]}}'
    json_response = task.get_json_response()
    assert isinstance(json_response, dict)
    assert json_response.get("key") == "value"
    assert isinstance(json_response.get("nested"), dict)
    assert json_response.get("nested", {}).get("array") == [1, 2, 3]

    # Test markdown code block parsing
    task.response = """```python
def test():
    return True
```
Some text
```json
{"key": "value"}
```"""
    json_response = task.get_json_response()
    assert json_response == {"key": "value"}

    markdown_response = task.get_markdown_response()
    assert isinstance(markdown_response, str)
    assert "```python" in markdown_response
    assert "```json" in markdown_response

    # Test markdown with replacements
    task.response = "**bold** *italic* [link](https://example.com)"
    markdown_response = task.get_markdown_response(
        replace_this="example.com", with_this="test.com"
    )
    assert isinstance(markdown_response, str)
    assert "test.com" in markdown_response
    assert "**bold**" in markdown_response

    # Test error responses
    mock_message = Message(
        id="msg_124",
        created_at=1234567890,
        thread_id="thread_123",
        role="assistant",
        content=[{"type": "text", "text": {"value": "Error: Invalid request"}}],
        file_ids=[],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={"error": True},
        status="completed",
        object="thread.message"
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]
    error_response = task.get_full_response()
    assert "Error:" in error_response

    # Test null responses
    task.response = None
    assert task.get_json_response() is None
    assert task.get_markdown_response() is None


def test_asmarkdown_function():
    """Test markdown processing functionality."""
    # Basic markdown processing
    test_string = "**bold** *italic*"
    result = asmarkdown(test_string)
    assert result == test_string

    # Test parameter handling with camelCase parameters
    result_with_replace = asmarkdown(test_string, replaceThis="bold", withThis="strong")
    assert result_with_replace == "**strong** *italic*"

    # Test None handling
    assert asmarkdown(None) is None

    # Test markdown with links
    test_string_with_link = "[test](https://example.com)"
    result_with_link = asmarkdown(test_string_with_link)
    assert result_with_link is not None
    assert isinstance(result_with_link, str)
    assert "[test]" in result_with_link


def test_celery_task_scheduling(
    mock_openai_client, mock_assistant, mock_thread, mock_run, mock_celery_task
):
    """Test core Celery task scheduling functionality.

    This test verifies:
    1. Basic task scheduling with completion callback
    2. Task status updates
    3. Callback scheduling on completion

    Does NOT test:
    - External tool execution
    - Specific callback implementations
    - Complex task workflows"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant", completionCall="test:callback")
    task.prompt = "Test task"
    run_id = task.create_run()

    mock_celery_task.delay.assert_called_once_with(
        run_id=run_id,
        thread_id=task.thread_id,
        completion_call="test:callback",
    )

    # Test task status updates
    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId=run_id,
        threadId=task.thread_id,
        assistant_id=task.assistant_id,
        status="completed",
        completion_call="test:callback",
    )

    # Verify callback scheduling
    mock_celery_task.delay.assert_called_with(
        run_id=run_id,
        thread_id=task.thread_id,
        completion_call="test:callback",
    )


def test_vision_file_handling(
    mock_openai_client, mock_assistant, mock_thread, mock_run
):
    """Test basic vision file support.

    This test verifies:
    1. Vision flag setting for image files
    2. Basic image file upload
    3. Vision message response handling

    Does NOT test:
    - Image processing
    - Complex vision scenarios
    - External vision services"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant")

    # Test image file upload
    mock_image = MagicMock()
    mock_image.name = "test.png"
    mock_image.read.return_value = b"image content"

    mock_openai_client.files.create.return_value = MagicMock(
        id="file_123", filename="test.png", purpose="assistants"
    )

    file_id = task.upload_file(file=mock_image)
    assert file_id == "file_123"
    assert task._fileids[-1]["vision"] is True

    # Test vision message response
    mock_message = Message(
        id="msg_123",
        thread_id="thread_123",
        role="assistant",
        content=[MagicMock(type="text", text=MagicMock(value="Image response"))],
        file_ids=["file_123"],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={},
        object="thread.message",
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]

    response = task.get_full_response()
    assert "Image response" in response


def test_completion_call_handling(
    mock_openai_client, mock_assistant, mock_thread, mock_run, mock_celery_task
):
    """Test basic completion call functionality.

    This test verifies:
    1. Completion call parameter storage
    2. Basic callback scheduling

    Does NOT test:
    - Callback implementations
    - External services
    - Complex callback scenarios"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant", completionCall="test:callback")
    assert task.completion_call == "test:callback"

    run_id = task.create_run()
    assert run_id == mock_run.id

    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId=run_id,
        threadId=task.thread_id,
        assistant_id=task.assistant_id,
        completion_call="test:callback",
        status="completed",
    )

    mock_celery_task.delay.assert_called_with(
        run_id=run_id, thread_id=task.thread_id, completion_call="test:callback"
    )


def test_file_message_handling(
    mock_openai_client, mock_assistant, mock_thread, mock_run
):
    """Test file-related message handling.

    This test verifies:
    1. File attachment in messages
    2. Response handling with files

    Does NOT test:
    - File upload (covered in test_file_upload)
    - File processing
    - External services"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant")
    task._fileids.append({"id": "file_123", "retrieval": True})

    # Test message with file attachment
    mock_message = Message(
        id="msg_123",
        thread_id="thread_123",
        role="assistant",
        content=[MagicMock(type="text", text=MagicMock(value="File response"))],
        file_ids=["file_123"],
        assistant_id="asst_123",
        run_id=mock_run.id,
        metadata={},
        object="thread.message",
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]

    response = task.get_full_response()
    assert "File response" in response


def test_api_key_validation(mock_openai_client):
    """Test basic API key validation.

    This test verifies:
    1. API key presence check
    2. Empty key validation

    Does NOT test:
    - Key format validation
    - Authentication flows
    - External API calls"""

    with pytest.raises(ValueError):
        assistantTask(assistantName="Test Assistant", api_key=None)

    with pytest.raises(ValueError):
        assistantTask(assistantName="Test Assistant", api_key="")


def test_task_isolation(mock_openai_client, mock_assistant):
    """Test task instance isolation.

    This test verifies:
    1. Metadata isolation
    2. Thread ID uniqueness

    Does NOT test:
    - User authentication
    - Permission models
    - External isolation"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]

    task1 = assistantTask(assistantName="Test Assistant", metadata={"key": "value1"})
    task2 = assistantTask(assistantName="Test Assistant", metadata={"key": "value2"})

    assert task1.metadata != task2.metadata
    assert task1.thread_id != task2.thread_id


def test_error_handling(mock_openai_client, mock_assistant, mock_thread):
    """Test basic error handling.

    This test verifies:
    1. Input validation errors
    2. Basic API error handling
    3. Status updates on errors

    Does NOT test:
    - Complex error scenarios
    - External service errors
    - Recovery mechanisms"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test input validation
    task = assistantTask(assistantName="Test Assistant")
    with pytest.raises(ValueError):
        task.get_all_messages()  # Missing thread ID

    with pytest.raises(ValueError):
        assistantTask(assistantName="Test Assistant", tools="invalid")

    # Test API error handling
    mock_openai_client.beta.assistants.list.return_value.data = []
    with pytest.raises(ValueError):
        assistantTask(assistantName="NonexistentAssistant")

    # Test error status updates
    mock_openai_client.beta.threads.runs.create.side_effect = openai.APIError(
        "API Error"
    )
    task = assistantTask(assistantName="Test Assistant")
    task.threadObject = mock_thread
    assert task.create_run() is None
    assert task.task.status == "failed"
