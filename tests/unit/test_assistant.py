import pytest
from django_openai_assistant.assistant import assistantTask, asmarkdown

def test_assistant_task_initialization():
    task = assistantTask(assistantName="Test Assistant")
    assert task.assistant_name == "Test Assistant"
    assert task.tools == []
    assert task.metadata == {}

def test_assistant_task_with_tools():
    tools = ["test_tool"]
    task = assistantTask(assistantName="Test Assistant", tools=tools)
    assert task.tools == tools

def test_asmarkdown_function():
    test_string = "**bold** *italic*"
    result = asmarkdown(test_string)
    assert result == test_string
