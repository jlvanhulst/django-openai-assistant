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

def test_tool_integration(mock_openai_client, mock_assistant, mock_thread, mock_run):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    
    # Test RAG search functionality
    class FuzzySearchParams(BaseModel):
        query: str
        results: int = 5
        pdf: bool = True
        period: str | None = None
        include_assessment: bool = False
        include_salesforce: bool = False
        only_unengaged: bool = False
    
    tool_call = ToolCall(
        id="call_123",
        type="function",
        function=MagicMock(
            name="fuzzy_search",
            arguments=json.dumps({
                "query": "AI companies in healthcare",
                "results": 3,
                "pdf": False,
                "period": "2023",
                "include_assessment": True
            })
        )
    )
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[tool_call])
    )
    mock_openai_client.beta.threads.runs.create.return_value = mock_run
    
    task = assistantTask(
        assistantName="Test Assistant",
        tools=["fuzzy_search", "create_and_send_query_pdf"],
        metadata={"user_email": "test@example.com"}
    )
    task.prompt = "Find healthcare AI companies"
    run_id = task.create_run()
    assert run_id == "run_123"
    assert task.status == "requires_action"
    
    # Test calendar event creation
    class CreateEventParams(BaseModel):
        email: str
        start: str
        end: str
        title: str
        description: str = ""
        attendees: list[str] = []
        address: str | None = None
        add_google_meet_link: bool = False
        calendar_id: str = "primary"
        time_zone: str = "America/New_York"
    
    # Test email processing
    class SendEmailParams(BaseModel):
        subject: str
        to: str
        body: str | None = None
        attachment: str | None = None
        
    tool_call = ToolCall(
        id="call_126",
        type="function",
        function=MagicMock(
            name="sendEmail",
            arguments=json.dumps({
                "subject": "Test Email",
                "to": "test@example.com",
                "body": "Test email body",
                "attachment": None
            })
        )
    )
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[tool_call])
    )
    
    task.prompt = "Send test email"
    run_id = task.create_run()
    assert run_id == "run_123"
    
    # Test RAG search with PDF generation
    tool_call = ToolCall(
        id="call_127",
        type="function",
        function=MagicMock(
            name="fuzzy_search",
            arguments=json.dumps({
                "query": "AI companies in fintech",
                "results": 5,
                "pdf": True,
                "period": "2024",
                "include_assessment": True,
                "include_salesforce": True
            })
        )
    )
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[tool_call])
    )
    
    task.prompt = "Generate PDF report for fintech AI companies"
    run_id = task.create_run()
    assert run_id == "run_123"
    
    # Test completion callback
    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
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
    assert task.task.status == "completed"
    assert task.completion_call == "test_callback"
    
    tool_call = ToolCall(
        id="call_123",
        type="function",
        function=MagicMock(
            name="create_event",
            arguments=json.dumps({
                "email": "test@example.com",
                "start": "2024-01-01T10:00:00",
                "end": "2024-01-01T11:00:00",
                "title": "Test Meeting",
                "description": "Test Description",
                "attendees": ["attendee@example.com"],
                "address": "123 Test St",
                "add_google_meet_link": True
            })
        )
    )
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[tool_call])
    )
    mock_openai_client.beta.threads.runs.create.return_value = mock_run
    
    task = assistantTask(
        assistantName="Test Assistant",
        tools=["create_event", "get_events", "add_location_to_calendar_event"],
        metadata={"user_email": "test@example.com"}
    )
    task.prompt = "Schedule a meeting"
    run_id = task.create_run()
    
    assert run_id == "run_123"
    assert task.status == "requires_action"
    
    # Test event retrieval
    class GetEventsParams(BaseModel):
        email: str
        calendar_id: str = "primary"
        date: str | None = None
        end_date: str | None = None
        timezone: str = "America/New_York"
    
    tool_call = ToolCall(
        id="call_124",
        type="function",
        function=MagicMock(
            name="get_events",
            arguments=json.dumps({
                "email": "test@example.com",
                "date": "2024-01-01",
                "timezone": "America/New_York"
            })
        )
    )
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[tool_call])
    )
    
    task.prompt = "Get today's meetings"
    run_id = task.create_run()
    assert run_id == "run_123"
    
    # Test location addition
    class AddLocationParams(BaseModel):
        user_email: str
        event_id: str
        new_location: str
        calendar_id: str = "primary"
    
    tool_call = ToolCall(
        id="call_125",
        type="function",
        function=MagicMock(
            name="add_location_to_calendar_event",
            arguments=json.dumps({
                "user_email": "test@example.com",
                "event_id": "event123",
                "new_location": "456 New Location"
            })
        )
    )
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[tool_call])
    )
    
    task.prompt = "Update meeting location"
    run_id = task.create_run()
    assert run_id == "run_123"
    
    # Test completion
    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId=run_id,
        threadId="thread_123",
        assistant_id="asst_123"
    )
    task.threadObject = mock_thread
    task.runObject = mock_run
    
    from django_openai_assistant.assistant import get_status
    status = get_status(f"{task.run_id},{task.thread_id}")
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

def test_metadata_management(mock_openai_client, mock_assistant, mock_thread, mock_run):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    
    # Test email metadata persistence
    email_metadata = {
        "from": "sender@example.com",
        "to": "recipient@example.com",
        "subject": "Test Subject",
        "thread_id": "thread_123",
        "msg_id": "msg_123"
    }
    task = assistantTask(
        assistantName="Test Assistant",
        metadata=email_metadata,
        completionCall="chatbot.google:replyToEmail"
    )
    assert task.metadata == email_metadata
    assert task.completion_call == "chatbot.google:replyToEmail"
    
    # Test Salesforce metadata
    sf_metadata = {
        "id": "001ABC",
        "sobject": "account",
        "company_name": "Test Company",
        "report_type": "quarterly"
    }
    task = assistantTask(
        assistantName="Test Assistant",
        metadata=sf_metadata,
        completionCall="chatbot.quickbooks:processReport"
    )
    assert task.metadata == sf_metadata
    
    # Test metadata persistence across runs
    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId="run_123",
        threadId="thread_123",
        assistant_id="asst_123",
        metadata=sf_metadata,
        completion_call="chatbot.quickbooks:processReport"
    )
    task.threadObject = mock_thread
    task.runObject = mock_run
    
    from django_openai_assistant.assistant import get_status
    status = get_status(f"{task.run_id},{task.thread_id}")
    assert task.task.metadata == sf_metadata
    assert task.completion_call == "chatbot.quickbooks:processReport"
    
    # Test metadata in tool calls
    tool_call = ToolCall(
        id="call_123",
        type="function",
        function=MagicMock(
            name="fuzzy_search",
            arguments=json.dumps({
                "query": "Test Company",
                "results": 3,
                "metadata": {"company_id": "001ABC"}
            })
        )
    )
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[tool_call])
    )
    task.prompt = "Search for company"
    run_id = task.create_run()
    assert run_id == "run_123"
    
    # Test metadata updates
    updated_metadata = sf_metadata.copy()
    updated_metadata["status"] = "processed"
    task.metadata = updated_metadata
    assert task.metadata["status"] == "processed"
    assert task.metadata["id"] == sf_metadata["id"]

def test_file_upload(mock_openai_client, mock_assistant):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.files.create.return_value = MagicMock(id="file_123")
    
    task = assistantTask(assistantName="Test Assistant")
    
    # Test text file upload with direct content
    text_content = b"test content"
    text_id = task.upload_file(
        file_content=text_content,
        filename="test.txt"
    )
    assert text_id == "file_123"
    assert len(task._fileids) == 1
    assert task._fileids[0]["vision"] is False
    assert task._fileids[0]["retrieval"] is True
    
    # Test PDF file upload with file object and explicit filename
    mock_pdf = MagicMock()
    mock_pdf.name = "original.pdf"
    mock_pdf.read.return_value = b"pdf content"
    pdf_id = task.upload_file(file=mock_pdf, filename="test.pdf")
    assert pdf_id == "file_123"
    assert len(task._fileids) == 2
    assert task._fileids[1]["vision"] is False
    assert task._fileids[1]["retrieval"] is True
    
    # Test image file upload with vision support
    mock_image = MagicMock()
    mock_image.name = "test.jpg"
    mock_image.read.return_value = b"image content"
    image_id = task.upload_file(file=mock_image)
    assert image_id == "file_123"
    assert len(task._fileids) == 3
    assert task._fileids[2]["vision"] is True
    assert task._fileids[2]["retrieval"] is False
    
    # Test file type detection based on filename
    mock_file = MagicMock()
    mock_file.name = "data.unknown"
    mock_file.read.return_value = b"content"
    file_id = task.upload_file(file=mock_file, filename="test.png")
    assert file_id == "file_123"
    assert len(task._fileids) == 4
    assert task._fileids[3]["vision"] is True
    assert task._fileids[3]["retrieval"] is False

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
    task.threadObject = mock_thread
    
    # Test streaming response
    mock_message = Message(
        id="msg_123",
        created_at=1234567890,
        thread_id="thread_123",
        role="assistant",
        content=[
            MagicMock(type="text", text=MagicMock(value="Part 1")),
            MagicMock(type="text", text=MagicMock(value="Part 2")),
            MagicMock(type="text", text=MagicMock(value="Part 3"))
        ],
        file_ids=[],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={},
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
        replace_this="example.com",
        with_this="test.com"
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
        content=[MagicMock(
            type="text",
            text=MagicMock(value="Error: Invalid request")
        )],
        file_ids=[],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={"error": True},
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
    test_string = "**bold** *italic*"
    result = asmarkdown(test_string)
    assert result == test_string
    
    result_with_replace = asmarkdown(
        test_string,
        replace_this="bold",
        with_this="strong"
    )
    assert result_with_replace == "**strong** *italic*"

def test_celery_integration(mock_openai_client, mock_assistant, mock_thread, mock_run, mock_celery_task):
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    
    # Test async task queuing
    task = assistantTask(
        assistantName="Test Assistant",
        metadata={"user_email": "test@example.com"},
        completionCall="chatbot.google:replyToEmail"
    )
    task.prompt = "Process this email"
    run_id = task.create_run()
    
    mock_celery_task.delay.assert_called_once_with(
        run_id=run_id,
        thread_id=task.thread_id,
        completion_call="chatbot.google:replyToEmail"
    )
    
    # Test task status monitoring
    mock_run.status = "in_progress"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId=run_id,
        threadId="thread_123",
        assistant_id="asst_123",
        status="in_progress"
    )
    task.threadObject = mock_thread
    task.runObject = mock_run
    
    from django_openai_assistant.assistant import get_status
    status = get_status(f"{task.run_id},{task.thread_id}")
    assert task.task.status == "in_progress"
    
    # Test completion callback
    mock_run.status = "completed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task.status = "completed"
    setattr(task.task, "completion_call", "chatbot.google:replyToEmail")
    task.task.save()
    
    status = get_status(f"{task.run_id},{task.thread_id}")
    mock_celery_task.delay.assert_called_with(
        run_id=run_id,
        thread_id=task.thread_id,
        completion_call="chatbot.google:replyToEmail"
    )
    
    # Test error handling in task
    mock_run.status = "failed"
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task.status = "failed"
    task.task.save()
    
    status = get_status(f"{task.run_id},{task.thread_id}")
    assert task.task.status == "failed"
    
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
        content_type="image/png"
    )
    
    file_id = task.upload_file(file=mock_image)
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
            MagicMock(
                type="image_file",
                image_file=MagicMock(file_id="file_123")
            ),
            MagicMock(
                type="text",
                text=MagicMock(value="The image shows a test pattern")
            )
        ],
        file_ids=["file_123"],
        assistant_id="asst_123",
        run_id="run_123",
        metadata={},
        object="thread.message"
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]
    
    task.prompt = "Analyze this image"
    run_id = task.create_run()
    assert run_id == mock_run.id
    
    response = task.get_full_response()
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
        content_type="image/jpeg"
    )
    
    file_id2 = task.upload_file(file=mock_image2)
    assert file_id2 == "file_456"
    assert task._fileids[-1]["vision"] is True
    
    # Test email with inline images
    mock_message.content = [
        MagicMock(
            type="text",
            text=MagicMock(value="Here are the analyzed images:")
        ),
        MagicMock(
            type="image_file",
            image_file=MagicMock(file_id="file_123")
        ),
        MagicMock(
            type="image_file",
            image_file=MagicMock(file_id="file_456")
        )
    ]
    mock_message.file_ids = ["file_123", "file_456"]
    
    response = task.get_full_response()
    assert "Here are the analyzed images" in response
    
    # Test image file deletion
    task.delete_file("file_123")
    mock_openai_client.files.delete.assert_called_with(file_id="file_123")
    
    # Test vision support with different file types
    image_types = [".png", ".jpg", ".jpeg", ".webp"]
    for ext in image_types:
        mock_file = MagicMock()
        mock_file.name = f"test{ext}"
        mock_file.read.return_value = b"image content"
        file_id = task.upload_file(file=mock_file)
        assert task._fileids[-1]["vision"] is True
        assert task._fileids[-1]["retrieval"] is False

def test_calendar_integration(mock_openai_client, mock_assistant, mock_thread, mock_run):
    """Test calendar integration based on vicbot/chatbot/calendar.py patterns"""
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    
    task = assistantTask(assistantName="Test Assistant")
    task.prompt = "Schedule a meeting with john@example.com tomorrow at 2pm"
    
    # Mock tool call for calendar event creation
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[
            ToolCall(
                id="call_123",
                type="function",
                function=MagicMock(
                    name="create_event",
                    arguments=json.dumps({
                        "email": "test@example.com",
                        "start": "2024-02-15T14:00:00",
                        "end": "2024-02-15T15:00:00",
                        "title": "Meeting with John",
                        "description": "Scheduled via assistant",
                        "attendees": ["john@example.com"],
                        "add_google_meet_link": True
                    })
                )
            )
        ])
    )
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    
    run_id = task.create_run()
    assert run_id == mock_run.id
    task.handle_tool_calls()
    
    # Test email integration with calendar invite
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[
            ToolCall(
                id="call_456",
                type="function",
                function=MagicMock(
                    name="send_email",
                    arguments=json.dumps({
                        "subject": "Meeting Invitation",
                        "to": "john@example.com",
                        "body": "I've scheduled a meeting for tomorrow at 2pm.",
                        "calendar_event_id": "evt_123"
                    })
                )
            )
        ])
    )
    task.handle_tool_calls()

def test_email_processing(mock_openai_client, mock_assistant, mock_thread, mock_run):
    """Test email processing based on vicbot/chatbot/google.py patterns"""
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    
    task = assistantTask(assistantName="Test Assistant")
    
    # Test email with attachments
    mock_file = MagicMock()
    mock_file.name = "report.pdf"
    mock_file.read.return_value = b"pdf content"
    
    mock_openai_client.files.create.return_value = MagicMock(
        id="file_123",
        filename="report.pdf",
        bytes=len(b"pdf content"),
        purpose="assistants"
    )
    
    file_id = task.upload_file(file=mock_file)
    task.prompt = "Send this report to john@example.com"
    
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[
            ToolCall(
                id="call_789",
                type="function",
                function=MagicMock(
                    name="send_email",
                    arguments=json.dumps({
                        "subject": "Report",
                        "to": "john@example.com",
                        "body": "Please find the report attached.",
                        "attachments": [{"file_id": file_id}]
                    })
                )
            )
        ])
    )
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    
    run_id = task.create_run()
    assert run_id == mock_run.id
    task.handle_tool_calls()

def test_rag_search_integration(mock_openai_client, mock_assistant, mock_thread, mock_run):
    """Test RAG search integration based on vicbot/chatbot/pinecone.py patterns"""
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    
    task = assistantTask(assistantName="Test Assistant")
    task.prompt = "Find companies working on AI infrastructure"
    
    # Mock RAG search tool call
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[
            ToolCall(
                id="call_321",
                type="function",
                function=MagicMock(
                    name="fuzzy_search",
                    arguments=json.dumps({
                        "query": "companies developing AI infrastructure and tooling",
                        "results": 5,
                        "pdf": True,
                        "include_assessment": True
                    })
                )
            )
        ])
    )
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    
    run_id = task.create_run()
    assert run_id == mock_run.id
    task.handle_tool_calls()
    
    # Test search result processing
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[
            ToolCall(
                id="call_654",
                type="function",
                function=MagicMock(
                    name="send_email",
                    arguments=json.dumps({
                        "subject": "AI Companies Research",
                        "to": "user@example.com",
                        "body": "Here are the companies I found...",
                        "attachments": [{"file_id": "file_789"}]
                    })
                )
            )
        ])
    )
    task.handle_tool_calls()

def test_file_upload_integration(mock_openai_client, mock_assistant, mock_thread, mock_run):
    """Test file upload integration based on vicbot/chatbot/upload.py patterns"""
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    
    task = assistantTask(assistantName="Test Assistant")
    
    # Test file upload with vision support
    mock_image = MagicMock()
    mock_image.name = "diagram.png"
    mock_image.read.return_value = b"image data"
    
    mock_openai_client.files.create.return_value = MagicMock(
        id="file_987",
        filename="diagram.png",
        bytes=len(b"image data"),
        purpose="assistants"
    )
    
    file_id = task.upload_file(file=mock_image)
    assert file_id == "file_987"
    
    # Test file processing with vision
    task.prompt = "Analyze this diagram"
    mock_run.status = "completed"
    mock_run.required_action = None
    mock_message = Message(
        id="msg_123",
        created_at=1234567890,
        thread_id="thread_123",
        role="assistant",
        content=[
            MagicMock(
                type="text",
                text=MagicMock(value="The diagram shows...")
            )
        ],
        file_ids=["file_987"],
        assistant_id="asst_123",
        run_id=mock_run.id,
        metadata={},
        object="thread.message"
    )
    mock_openai_client.beta.threads.messages.list.return_value.data = [mock_message]
    
    run_id = task.create_run()
    assert run_id == mock_run.id
    response = task.get_full_response()
    assert "The diagram shows" in response

def test_api_key_security(mock_openai_client, mock_assistant, mock_thread, mock_run):
    """Test API key management and security"""
    # Test API key validation
    mock_openai_client.beta.assistants.list.side_effect = Exception("Invalid API key")
    task = assistantTask(assistantName="Test Assistant")
    with pytest.raises(Exception, match="Invalid API key"):
        task.create_run()
    
    # Test API key rotation
    mock_openai_client.beta.assistants.list.side_effect = None
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    task = assistantTask(assistantName="Test Assistant", api_key="new_key")
    assert task.client.api_key == "new_key"

def test_file_permissions(mock_openai_client, mock_assistant, mock_thread, mock_run):
    """Test file access permissions and security"""
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    
    task = assistantTask(assistantName="Test Assistant")
    
    # Test file upload permissions
    mock_file = MagicMock()
    mock_file.name = "sensitive.pdf"
    mock_file.read.return_value = b"sensitive content"
    
    # Test file purpose validation
    mock_openai_client.files.create.side_effect = Exception("Invalid file purpose")
    with pytest.raises(Exception, match="Invalid file purpose"):
        task.upload_file(file=mock_file, purpose="invalid")
    
    # Test file access scoping
    mock_openai_client.files.create.side_effect = None
    mock_openai_client.files.create.return_value = MagicMock(
        id="file_123",
        filename="sensitive.pdf",
        purpose="assistants"
    )
    file_id = task.upload_file(file=mock_file)
    
    # Test file deletion
    mock_openai_client.files.delete.return_value = True
    assert task.delete_file(file_id) is True

def test_user_context_isolation(mock_openai_client, mock_assistant, mock_thread, mock_run):
    """Test user context isolation and security boundaries"""
    mock_openai_client.beta.assistants.list.return_value.data = [mock_assistant]
    mock_openai_client.beta.threads.create.return_value = mock_thread
    
    # Test user context separation
    task1 = assistantTask(assistantName="Test Assistant", metadata={"user_id": "user1"})
    task2 = assistantTask(assistantName="Test Assistant", metadata={"user_id": "user2"})
    
    # Verify thread isolation
    assert task1.thread_id != task2.thread_id
    
    # Test metadata isolation
    task1.set_metadata({"sensitive": "data1"})
    task2.set_metadata({"sensitive": "data2"})
    assert task1.metadata.get("sensitive") != task2.metadata.get("sensitive")
    
    # Test file isolation
    mock_file = MagicMock()
    mock_file.name = "user1_data.pdf"
    mock_openai_client.files.create.return_value = MagicMock(
        id="file_123",
        filename="user1_data.pdf",
        purpose="assistants"
    )
    file_id = task1.upload_file(file=mock_file)
    
    # Verify file access boundaries
    mock_openai_client.files.retrieve.side_effect = Exception("Access denied")
    with pytest.raises(Exception, match="Access denied"):
        task2.retrieve_file(file_id)

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
    
    # Test API rate limiting
    mock_openai_client.beta.threads.runs.create.side_effect = openai.RateLimitError(
        "Rate limit exceeded",
        response=MagicMock(status_code=429)
    )
    task = assistantTask(assistantName="Test Assistant")
    task.threadObject = mock_thread
    assert task.create_run() is None
    assert task.task.status == "failed"
    
    # Test API timeout
    mock_openai_client.beta.threads.runs.create.side_effect = openai.APITimeoutError(
        "Request timed out"
    )
    assert task.create_run() is None
    assert task.task.status == "failed"
    
    # Test API connection error
    mock_openai_client.beta.threads.runs.create.side_effect = openai.APIConnectionError(
        "Connection error"
    )
    assert task.create_run() is None
    assert task.task.status == "failed"
    
    # Test invalid file upload
    mock_file = MagicMock()
    mock_file.name = "test.exe"
    mock_file.read.return_value = b"invalid content"
    mock_openai_client.files.create.side_effect = openai.BadRequestError(
        "Unsupported file type",
        response=MagicMock(status_code=400)
    )
    with pytest.raises(ValueError, match="File upload failed"):
        task.upload_file(file=mock_file)
    
    # Test tool call errors
    mock_run.status = "requires_action"
    mock_run.required_action = MagicMock(
        submit_tool_outputs=MagicMock(tool_calls=[
            ToolCall(
                id="call_123",
                type="function",
                function=MagicMock(
                    name="invalid_function",
                    arguments="{}"
                )
            )
        ])
    )
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task = OpenaiTask.objects.create(
        runId="run_123",
        threadId="thread_123",
        assistant_id="asst_123",
        status="requires_action"
    )
    task.runObject = mock_run
    
    # Test tool call timeout
    mock_openai_client.beta.threads.runs.submit_tool_outputs.side_effect = openai.APITimeoutError(
        "Tool call timed out"
    )
    task.handle_tool_calls()
    assert task.task.status == "failed"
    
    # Test run status handling
    mock_run.status = "failed"
    mock_run.last_error = MagicMock(code="rate_limit_exceeded")
    mock_openai_client.beta.threads.runs.retrieve.return_value = mock_run
    task.task.status = "failed"
    task.task.save()
    
    from django_openai_assistant.assistant import get_status
    status = get_status(f"{task.run_id},{task.thread_id}")
    assert task.task.status == "failed"
    assert task.task.error_code == "rate_limit_exceeded"
