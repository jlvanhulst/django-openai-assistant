import json
from unittest.mock import MagicMock, patch

import pytest
from django.utils import timezone
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Run, Message
from openai.types.beta.threads.runs import ToolCall
from pydantic import BaseModel

from django_openai_assistant.assistant import (
    assistantTask,
    asmarkdown,
    set_default_tools,
    get_assistant,
    create_assistant
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
        object="assistant"
    )

@pytest.fixture
def mock_thread():
    return Thread(
        id="thread_123",
        created_at=1234567890,
        metadata={},
        object="thread"
    )

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
        instructions=None,
        tools=[],
        file_ids=[],
        metadata={},
        object="thread.run"
    )

def test_assistant_creation_and_configuration(mock_openai_client, mock_assistant, mock_thread, mock_run):
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
        completionCall="test:callback"
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
        metadata={"user": "test@example.com"}
    )
    assert task.completion_call == "test_callback"
    assert task.metadata == {"user": "test@example.com"}
    
    # Test assistant creation with tools
    tools = {
        "fuzzy_search": {"module": "pinecone"},
        "create_event": {"module": "calendar"}
    }
    new_assistant = create_assistant(
        name="New Assistant",
        tools=list(tools.keys()),
        model="gpt-4",
        instructions="Test assistant instructions"
    )
    assert new_assistant.id == "asst_123"
    assert new_assistant.model == "gpt-4"
    
    # Test getting existing assistants
    assistants = get_assistant()
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
        "create_event": {"module": "calendar"}
    }
    result = set_default_tools(tools=list(tools.keys()), package="chatbot")
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
    run_id = task.create_run(temperature=0.7)
    
    assert run_id == "run_123"
    assert task.threadObject and task.threadObject.id == "thread_123"
    assert task.task.runId == "run_123"

def test_tool_calling_and_completion(mock_openai_client, mock_assistant, mock_thread, mock_run):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    
    # Test completion callback functionality
    task = assistantTask(
        assistantName="Test Assistant",
        completionCall="test:callback",
        metadata={"id": "test123"}
    )
    assert task.completion_call == "test:callback"
    
    # Test tool calling with completion
    mock_openai_client.beta.threads.create.return_value = mock_thread
    
    # Test tool calling with Pydantic model
    class TestParams(BaseModel):
        company_name: str
        email: str | None = None
    
    tool_call = ToolCall(
        id="call_123",
        type="function",
        function=MagicMock(
            name="getCompany",
            arguments='{"company_name": "Test Corp", "email": "test@example.com"}'
        )
    )
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[tool_call])
    )
    mock_openai_client.beta.threads.runs.create.return_value = mock_run
    
    # Test with real-world tool configuration
    tools = {
        "getCompany": {"module": "salesforce"},
        "fuzzy_search": {"module": "pinecone"}
    }
    task = assistantTask(
        assistantName="Test Assistant",
        tools=list(tools.keys()),
        metadata={"user_email": "test@example.com"}
    )
    task.prompt = "Get company information"
    run_id = task.create_run()
    
    assert run_id == "run_123"
    assert task.status == "requires_action"
    assert task.metadata["user_email"] == "test@example.com"
    
    # Test completion callback
    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    
    task = assistantTask(
        assistantName="Test Assistant",
        completionCall="test_callback"
    )
    task.task = OpenaiTask.objects.create(
        runId=run_id,
        threadId="thread_123",
        assistant_id="asst_123",
        completion_call="test_callback"
    )
    task.threadObject = mock_thread
    task.runObject = mock_run
    
    from django_openai_assistant.assistant import get_status
    status = get_status(f"{task.run_id},{task.thread_id}")
    assert status == task.response
    assert task.task.status == "completed"

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
            MagicMock(type="text", text=MagicMock(value="Text response")),
            MagicMock(
                type="image_file",
                image_file=MagicMock(file_id="file_123")
            )
        ],
        file_ids=["file_123"],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={},
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
            completionCall="test_callback" if status == "completed" else None
        )
        task.task = OpenaiTask.objects.create(
            runId="run_123",
            threadId="thread_123",
            assistant_id="asst_123",
            completion_call="test_callback" if status == "completed" else None
        )
        task.threadObject = mock_thread
        task.runObject = mock_run
        
        from django_openai_assistant.assistant import get_status
        current_status = get_status(f"{task.run_id},{task.thread_id}")
        assert task.task.status == status
        
        if status == "completed":
            assert task.completion_call == "test_callback"
        elif status == "requires_action":
            assert task.task.status == "requires_action"

def test_file_upload(mock_openai_client, mock_assistant):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.files.create.return_value = MagicMock(id="file_123")
    
    task = assistantTask(assistantName="Test Assistant")
    file_content = b"test content"
    file_id = task.upload_file(
        file_content=file_content,
        filename="test.txt"
    )
    
    assert file_id == "file_123"
    assert len(task._fileids) == 1
    assert task._fileids[0]["vision"] is False
    assert task._fileids[0]["retrieval"] is True

def test_vision_file_upload(mock_openai_client, mock_assistant):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.files.create.return_value = MagicMock(id="file_123")
    
    task = assistantTask(assistantName="Test Assistant")
    file_content = b"test image"
    file_id = task.upload_file(
        file_content=file_content,
        filename="test.jpg"
    )
    
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
        content=[MagicMock(type="text", text=MagicMock(value="Test response"))],
        file_ids=[],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={},
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
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread

    task = assistantTask(assistantName="Test Assistant")
    
    # Test JSON response parsing
    task.response = '{"key": "value"}'
    json_response = task.get_json_response()
    assert json_response == {"key": "value"}
    
    # Test markdown code block parsing
    task.response = "```json\n{\"key\": \"value\"}\n```"
    json_response = task.get_json_response()
    assert json_response == {"key": "value"}
    
    # Test markdown with replacements
    task.response = "**bold** *italic*"
    markdown_response = task.get_markdown_response(
        replaceThis="bold",
        withThis="strong"
    )
    assert markdown_response == "**strong** *italic*"
    
    # Test null responses
    task.response = None
    assert task.get_json_response() is None
    assert task.get_markdown_response() is None

def test_asmarkdown_function():
    test_string = "**bold** *italic*"
    result = asmarkdown(test_string)
    assert result == test_string
    
    result_with_replace = asmarkdown(
        test_string,
        replaceThis="bold",
        withThis="strong"
    )
    assert result_with_replace == "**strong** *italic*"

def test_error_handling(mock_openai_client, mock_assistant, mock_thread, mock_run):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    
    # Test missing thread ID
    task = assistantTask(assistantName="Test Assistant")
    with pytest.raises(ValueError, match="Thread ID is required"):
        task.get_all_messages()
    
    # Test invalid assistant name
    mock_openai_client.beta.assistants.list.return_value.data = []
    with pytest.raises(ValueError, match="Assistant .* not found"):
        assistantTask(assistantName="NonexistentAssistant")
    
    # Test invalid tool configuration
    with pytest.raises(ValueError, match="tools must be.*"):
        assistantTask(assistantName="Test Assistant", tools="invalid")
    
    # Test run creation failure
    mock_openai_client.beta.threads.runs.create.side_effect = Exception("API Error")
    task = assistantTask(assistantName="Test Assistant")
    task.threadObject = mock_thread
    assert task.create_run() is None
    
    # Test run status handling
    mock_run.status = "failed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId="run_123",
        threadId="thread_123",
        assistant_id="asst_123"
    )
    task.runObject = mock_run
    from django_openai_assistant.assistant import get_status
    status = get_status(f"{task.run_id},{task.thread_id}")
    assert task.task.status == "failed"
