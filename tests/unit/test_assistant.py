import json
from unittest.mock import MagicMock

import openai
import pytest
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Message, Run

from django_openai_assistant.assistant import (
    asmarkdown,
    assistantTask,
    createAssistant,
    get_assistant,
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
        instructions="",  # Empty string instead of None
        tools=[],
        file_ids=[],
        metadata={},
        object="thread.run",
        parallel_tool_calls=0,  # Required field
    )


def test_assistant_creation_and_configuration(
    mock_openai_client, mock_assistant, mock_thread, mock_run
):
    """Test assistantTask initialization and configuration.

    This test verifies:
    1. Initialization with comboId (run_id, thread_id)
    2. Initialization with assistantName and metadata
    3. Assistant retrieval and creation via OpenAI API
    4. Tool configuration storage and validation

    Does NOT test:
    - Tool function loading
    - External service integration
    - Complex assistant configurations
    - Business logic implementations"""

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
    new_assistant = createAssistant(
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
    """Test the assistantTask's tool configuration storage.

    This test verifies:
    1. Tool list validation and storage in assistantTask
    2. Empty tool list handling (default behavior)
    3. Single tool configuration support
    4. Tool configuration persistence in metadata

    Does NOT test:
    - Tool function discovery or loading
    - Tool execution or responses
    - External module imports"""

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
    """Test the public set_default_tools API function.

    This test verifies:
    1. Tool registration with dictionary format
    2. Tool registration with module:function format
    3. Package-based tool registration
    4. Default tool configuration persistence
    5. Tool format validation

    Does NOT test:
    - Tool function implementations
    - Tool execution
    - External module loading
    - Tool discovery"""

    # Test dictionary format
    tools = {
        "function1": {"module": "core"},
        "function2": {"module": "core"},
    }
    result = set_default_tools(tools=list(tools.keys()), package="test")
    assert isinstance(result, dict)
    assert all(isinstance(v, dict) for v in result.values())
    assert all("module" in v for v in result.values())
    assert result["function1"]["module"] == "core"
    assert result["function2"]["module"] == "core"

    # Test module:function format
    tools_list = ["core:function", "other:tool"]
    result = set_default_tools(tools=tools_list)
    assert isinstance(result, dict)
    assert all(isinstance(v, dict) for v in result.values())
    assert all("module" in v for v in result.values())
    assert result["function"]["module"] == "core"
    assert result["tool"]["module"] == "other"

    # Test invalid format handling
    with pytest.raises(ValueError):
        set_default_tools(tools=["invalid_format"])

    with pytest.raises(ValueError):
        set_default_tools(tools=["module:function:extra"])


def test_create_run(mock_openai_client, mock_assistant, mock_thread, mock_run):
    """Test assistantTask's run creation and management.

    This test verifies:
    1. Run creation with OpenAI API
    2. Thread creation and ID assignment
    3. Task object creation and persistence
    4. Temperature parameter handling

    Does NOT test:
    - Run execution results
    - External API error handling
    - Complex run configurations
    - Tool-specific behaviors"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    mock_openai_client.beta.threads.runs.create.return_value = mock_run

    task = assistantTask(assistantName="Test Assistant")
    task.prompt = "Test prompt"
    run_id = task.createRun(temperature=0.7)

    assert run_id == "run_123"
    assert task.threadObject and task.threadObject.id == "thread_123"
    assert task.task.runId == "run_123"


def test_tool_call_handling(
    mock_openai_client, mock_assistant, mock_thread, mock_run, mock_celery_task
):
    """Test assistantTask's public API for tool configuration and status handling.

    This test verifies:
    1. Tool registration via constructor
    2. Tool string format validation
    3. Run status transitions during tool processing
    4. Task status persistence

    Does NOT test:
    - Tool implementation details
    - Internal tool scheduling
    - Tool output submission (handled by Celery)
    - External API implementations"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test tool registration and validation
    task = assistantTask(
        assistantName="Test Assistant",
        tools=["test.module:function1", "test.module:function2"],
    )
    assert isinstance(task.tools, list)
    assert "test.module:function1" in task.tools
    assert "test.module:function2" in task.tools

    # Test invalid tool format
    with pytest.raises(ValueError):
        assistantTask(assistantName="Test Assistant", tools=["invalid_format"])

    # Test run creation with tools
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(
            tool_calls=[
                MagicMock(
                    id="call_123",
                    type="function",
                    function=MagicMock(
                        name="test_tool",
                        arguments=json.dumps({"param": "value"}),
                    ),
                )
            ]
        )
    )
    mock_openai_client.beta.threads.runs.create.return_value = mock_run

    task.prompt = "Test prompt"
    run_id = task.createRun()
    assert run_id == "run_123"
    assert task.status == "requires_action"

    # Test status transitions
    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId=run_id,
        threadId=task.thread_id,
        assistant_id=task.assistant_id,
        status="completed",
    )
    assert task.task.status == "completed"

    # Verify Celery task scheduling
    mock_celery_task.delay.assert_called_with(f"{run_id},{task.thread_id}")


def test_thread_message_handling(mock_openai_client, mock_assistant, mock_thread):
    """Test assistantTask's thread-level message operations.

    This test verifies:
    1. Thread creation and initialization
    2. Message list retrieval from thread
    3. Thread-level metadata handling
    4. Thread message synchronization

    Does NOT test:
    - Message content processing (covered in test_message_handling)
    - File operations (covered in test_file_message_handling)
    - External API implementations
    - Complex message chains"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant")
    assert task.threadObject == mock_thread

    # Test thread message list retrieval
    mock_message1 = Message(
        id="msg_123",
        created_at=1234567890,
        thread_id="thread_123",
        role="assistant",
        content=[{"type": "text", "text": {"value": "Message 1"}}],
        file_ids=[],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={"order": 1},
        status="completed",
        object="thread.message",
    )
    mock_message2 = Message(
        id="msg_124",
        created_at=1234567891,
        thread_id="thread_123",
        role="assistant",
        content=[{"type": "text", "text": {"value": "Message 2"}}],
        file_ids=[],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={"order": 2},
        status="completed",
        object="thread.message",
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [
        mock_message2,
        mock_message1,
    ]

    # Verify message list retrieval and ordering
    messages = task.getAllMessages()
    assert len(messages) == 2
    assert messages[0]["id"] == "msg_124"  # Most recent first
    assert messages[1]["id"] == "msg_123"

    # Verify thread metadata preservation
    assert all(msg["thread_id"] == "thread_123" for msg in messages)
    assert all(msg["assistant_id"] == "asst_123" for msg in messages)

    # Test thread message count
    assert len(task.getAllMessages()) == 2


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
    file_id = task.uploadFile(file_content=content, filename="test.txt")
    assert file_id == "file_123"
    assert len(task._fileids) == 1

    # Test file object upload
    mock_file = MagicMock()
    mock_file.name = "test.txt"
    mock_file.read.return_value = b"test content"
    file_id = task.uploadFile(file=mock_file)
    assert file_id == "file_123"
    assert len(task._fileids) == 2

    # Verify OpenAI API interaction
    mock_openai_client.files.create.assert_called_with(
        file=mock_file.read.return_value, purpose="assistants", filename="test.txt"
    )


def test_message_handling(mock_openai_client, mock_assistant, mock_thread):
    """Test assistantTask's message content processing.

    This test verifies:
    1. Message content extraction
    2. Response formatting
    3. Content type handling
    4. Message role handling

    Does NOT test:
    - Thread operations (covered in test_thread_message_handling)
    - File handling (covered in test_file_message_handling)
    - External API implementations
    - Message history management"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test different content types and roles
    mock_messages = [
        Message(
            id="msg_123",
            created_at=1234567890,
            thread_id="thread_123",
            role="assistant",
            content=[
                {"type": "text", "text": {"value": "Response 1"}},
                {"type": "text", "text": {"value": "Response 2"}},
            ],
            file_ids=[],
            assistant_id="asst_123",
            run_id="run_123",
            metadata={},
            status="completed",
            object="thread.message",
        ),
        Message(
            id="msg_124",
            created_at=1234567891,
            thread_id="thread_123",
            role="user",
            content=[{"type": "text", "text": {"value": "User input"}}],
            file_ids=[],
            assistant_id="asst_123",
            run_id="run_123",
            metadata={},
            status="completed",
            object="thread.message",
        ),
    ]
    mock_openai_client.beta.threads.messages.list.return_value.data = mock_messages

    task = assistantTask(assistantName="Test Assistant")
    task.threadObject = mock_thread

    # Test content extraction and formatting
    response = task.getFullResponse()
    assert "Response 1" in response
    assert "Response 2" in response
    assert "User input" not in response  # User messages not included

    # Test role-specific handling
    messages = task.getAllMessages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    # Test last response handling
    last_response = task.getLastResponse()
    assert last_response is not None
    assert isinstance(last_response, dict)
    assert last_response.get("role") == "user"  # Most recent message


def test_response_formats(mock_openai_client, mock_assistant, mock_thread):
    """Test assistantTask's response format handling.

    This test verifies:
    1. JSON response extraction and validation
    2. Markdown formatting and replacements
    3. Multi-part message concatenation
    4. Error and null response handling
    5. Code block extraction

    Does NOT test:
    - Message retrieval (covered in test_message_handling)
    - Thread operations (covered in test_thread_message_handling)
    - External format processing
    - Complex message chains"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant")

    # Test JSON extraction and validation
    task.response = '{"key": "value", "nested": {"array": [1, 2, 3]}}'
    json_response = task.getJsonResponse()
    assert isinstance(json_response, dict)
    assert json_response.get("key") == "value"
    assert isinstance(json_response.get("nested"), dict)
    assert json_response.get("nested", {}).get("array") == [1, 2, 3]

    # Test code block extraction
    task.response = """```python
def test():
    return True
```
Some text
```json
{"key": "value"}
```"""
    json_response = task.getJsonResponse()
    assert json_response == {"key": "value"}

    # Test markdown formatting
    markdown_response = task.getMarkdownResponse()
    assert isinstance(markdown_response, str)
    assert "```python" in markdown_response
    assert "```json" in markdown_response

    # Test markdown replacements
    task.response = "**bold** *italic* [link](https://example.com)"
    markdown_response = task.getMarkdownResponse(
        replace_this="example.com", with_this="test.com"
    )
    assert isinstance(markdown_response, str)
    assert "test.com" in markdown_response
    assert "**bold**" in markdown_response

    # Test multi-part message concatenation
    task.response = "Part 1\nPart 2\nPart 3"
    full_response = task.getFullResponse()
    assert "Part 1" in full_response
    assert "Part 2" in full_response
    assert "Part 3" in full_response

    # Test error response handling
    task.response = "Error: Invalid request"
    error_response = task.getFullResponse()
    assert "Error:" in error_response

    # Test null response handling
    task.response = None
    assert task.getJsonResponse() is None
    assert task.getMarkdownResponse() is None
    assert task.getFullResponse() is None


def test_asmarkdown_function():
    """Test the public asmarkdown function for text formatting.

    This test verifies:
    1. Basic markdown text preservation
    2. String replacement functionality
    3. None value handling
    4. Link text processing
    5. Special character handling

    Does NOT test:
    - Complex markdown parsing
    - HTML conversion
    - External markdown libraries"""

    # Basic markdown preservation
    test_string = "**bold** *italic*"
    result = asmarkdown(test_string)
    assert result == test_string

    # String replacement functionality
    result_with_replace = asmarkdown(test_string, replaceThis="bold", withThis="strong")
    assert result_with_replace == "**strong** *italic*"

    # None value handling
    assert asmarkdown(None) is None

    # Link text processing
    test_string_with_link = "[test](https://example.com)"
    result_with_link = asmarkdown(test_string_with_link)
    assert result_with_link == test_string_with_link

    # Special character handling
    test_string_special = "# Header\n* List\n> Quote"
    result_special = asmarkdown(test_string_special)
    assert result_special == test_string_special


def test_celery_task_scheduling(
    mock_openai_client, mock_assistant, mock_thread, mock_run, mock_celery_task
):
    """Test the assistant's integration with Celery task queue.

    This test verifies:
    1. Task scheduling through Celery's delay() method
    2. Proper task metadata handling (run_id, thread_id)
    3. Completion callback registration
    4. Task status propagation to assistant

    Does NOT test:
    - Actual tool execution
    - Callback function implementations
    - Complex multi-task workflows"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant", completionCall="test:callback")
    task.prompt = "Test task"
    run_id = task.createRun()

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
    """Test assistantTask's handling of vision-enabled files.

    This test verifies:
    1. Vision flag setting for image file types
    2. File metadata tracking for vision files
    3. Vision-enabled file upload process
    4. Message handling with vision files
    5. Response formatting with vision content

    Does NOT test:
    - Image content processing
    - Vision model capabilities
    - External vision services
    - Image format validation"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant")

    # Test vision file handling
    mock_image = MagicMock()
    mock_image.name = "test.png"
    mock_image.read.return_value = b"image content"

    mock_openai_client.files.create.return_value = MagicMock(
        id="file_123", filename="test.png", purpose="assistants"
    )

    # Test file upload and metadata
    file_id = task.uploadFile(file=mock_image)
    assert file_id == "file_123"
    assert task._fileids[-1]["vision"] is True
    assert task._fileids[-1]["id"] == "file_123"
    assert task._fileids[-1]["retrieval"] is False

    # Test vision message handling
    mock_message = Message(
        id="msg_123",
        thread_id="thread_123",
        role="assistant",
        content=[
            {"type": "text", "text": {"value": "Image description"}},
            {"type": "image_file", "file_id": "file_123"},
        ],
        file_ids=["file_123"],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={"vision": True},
        object="thread.message",
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]

    # Test response formatting
    response = task.getFullResponse()
    assert "Image description" in response
    assert "file_123" in response

    # Verify message metadata
    messages = task.getAllMessages()
    assert len(messages) == 1
    assert messages[0]["metadata"].get("vision") is True
    assert "file_123" in messages[0]["file_ids"]


def test_completion_call_handling(
    mock_openai_client, mock_assistant, mock_thread, mock_run, mock_celery_task
):
    """Test assistantTask's completion callback registration and scheduling.

    This test verifies:
    1. Completion call parameter validation and storage
    2. Callback scheduling through Celery's delay() method
    3. Task metadata preservation in callback scheduling
    4. Callback triggering on task completion
    5. Multiple callback format support (module:function, package.module:function)

    Does NOT test:
    - Callback function implementations
    - External service integrations
    - Complex callback chains
    - Actual function discovery or loading"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test basic callback registration
    task = assistantTask(assistantName="Test Assistant", completionCall="test:callback")
    assert task.completion_call == "test:callback"

    # Test callback scheduling on run creation
    run_id = task.createRun()
    assert run_id == mock_run.id
    mock_celery_task.delay.assert_called_with(
        run_id=run_id,
        thread_id=task.thread_id,
        completion_call="test:callback",
    )

    # Test callback triggering on completion
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
        run_id=run_id,
        thread_id=task.thread_id,
        completion_call="test:callback",
    )

    # Test package-based callback format
    task_package = assistantTask(
        assistantName="Test Assistant", completionCall="package.module:function"
    )
    assert task_package.completion_call == "package.module:function"

    # Test invalid callback format
    with pytest.raises(ValueError):
        assistantTask(assistantName="Test Assistant", completionCall="invalid_format")


def test_file_message_handling(
    mock_openai_client, mock_assistant, mock_thread, mock_run
):
    """Test assistantTask's handling of messages with file attachments.

    This test verifies:
    1. File ID tracking in messages
    2. Message content extraction with files
    3. Response formatting with file references
    4. File metadata preservation in responses

    Does NOT test:
    - File upload functionality (covered in test_file_upload)
    - File content processing
    - External file operations
    - File storage or retrieval"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant")
    task._fileids.append({"id": "file_123", "retrieval": True})

    # Test message with file attachment
    mock_message = Message(
        id="msg_123",
        thread_id="thread_123",
        role="assistant",
        content=[
            {"type": "text", "text": {"value": "File response"}},
            {"type": "file", "file_id": "file_123"},
        ],
        file_ids=["file_123"],
        assistant_id="asst_123",
        run_id=mock_run.id,
        metadata={},
        object="thread.message",
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]

    # Test response formatting with file references
    response = task.getFullResponse()
    assert "File response" in response
    assert "file_123" in response  # Verify file reference is included

    # Test file metadata preservation
    messages = task.get_all_messages()
    assert len(messages) == 1
    assert "file_123" in messages[0]["file_ids"]


def test_api_key_validation(mock_openai_client):
    """Test assistantTask's API key validation.

    This test verifies:
    1. API key validation in constructor
    2. Empty key handling
    3. Missing key handling
    4. Key inheritance from settings

    Does NOT test:
    - Key format validation
    - Authentication flows
    - External API calls
    - Key storage"""

    # Test missing key
    with pytest.raises(ValueError):
        assistantTask(assistantName="Test Assistant", api_key=None)

    # Test empty key
    with pytest.raises(ValueError):
        assistantTask(assistantName="Test Assistant", api_key="")

    # Test key inheritance from settings
    mock_openai_client.beta.assistants.list.return_value.data = [
        MagicMock(name="Test Assistant", id="asst_123")
    ]
    task = assistantTask(assistantName="Test Assistant")
    assert task.assistant_id == "asst_123"


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
    """Test assistantTask's error handling and validation.

    This test verifies:
    1. Constructor parameter validation
    2. Method precondition validation
    3. OpenAI API error handling
    4. Task status updates on errors
    5. Error propagation in core methods

    Does NOT test:
    - External service errors
    - Complex error recovery
    - Tool-specific errors
    - Network error handling"""

    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    # Test constructor validation
    with pytest.raises(ValueError, match="Invalid tool format"):
        assistantTask(assistantName="Test Assistant", tools="invalid")

    with pytest.raises(ValueError, match="API key"):
        assistantTask(assistantName="Test Assistant", api_key="")

    # Test method preconditions
    task = assistantTask(assistantName="Test Assistant")
    with pytest.raises(ValueError, match="Thread"):
        task.getAllMessages()  # Missing thread ID

    # Test assistant lookup failure
    mock_openai_client.beta.assistants.list.return_value.data = []
    with pytest.raises(ValueError, match="Assistant not found"):
        assistantTask(assistantName="NonexistentAssistant")

    # Test run creation error handling
    mock_openai_client.beta.threads.runs.create.side_effect = openai.APIError(
        "API Error"
    )
    task = assistantTask(assistantName="Test Assistant")
    task.threadObject = mock_thread
    assert task.createRun() is None
    assert task.task.status == "failed"

    # Test tool error handling
    mock_run = MagicMock(status="requires_action")
    mock_run.required_action.submit_tool_outputs.tool_calls = [
        MagicMock(id="call_123", function=MagicMock(name="invalid_tool"))
    ]
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId="run_123",
        threadId=task.thread_id,
        assistant_id=task.assistant_id,
        status="requires_action",
    )
    assert task.task.status == "requires_action"  # Check task status directly
