import importlib
import inspect
import json
from typing import Any, BinaryIO, Callable, Optional, Union

import markdown
from celery import chord, shared_task
from django.conf import settings
from django.utils import timezone
from openai import OpenAI
from pydantic import BaseModel

from .models import OpenaiTask

_DEFAULT_TOOLS = {}
_PACKAGE = None


def set_default_tools(tools: Optional[list] = None, package: Optional[str] = None):
    """
    Set the default tools to use for the assistant.

    This is a global setting used for all assistants if no tools array is provided.

    Parameters:
        tools: List of strings in format "<module>:<function>"
            Examples:
            - ["salesforce:getCompany"]
            - ["chatbot.salesforce:getCompany"]
        package: Optional package prefix for tool modules
            Used when module names don't include package

    Behavior:
        - If _DEFAULT_TOOLS is already a dictionary, add new tools to it.
        - If tools is a list, convert it to a dictionary and merge.
        - If tools is None, do nothing.

    Returns:
        The updated _DEFAULT_TOOLS dictionary.
    """
    global _DEFAULT_TOOLS, _PACKAGE
    if tools is not None:
        if isinstance(tools, list):
            # Convert list of "module:function" to dict
            new_tools = {
                func: {"module": mod}
                for mod, func in (tool.split(":") for tool in tools)
            }
            if isinstance(_DEFAULT_TOOLS, dict):
                # Merge new_tools into _DEFAULT_TOOLS
                _DEFAULT_TOOLS.update(new_tools)
            else:
                # Initialize _DEFAULT_TOOLS as new_tools
                _DEFAULT_TOOLS = new_tools
        elif isinstance(tools, dict):
            if isinstance(_DEFAULT_TOOLS, dict):
                # Merge incoming dict into _DEFAULT_TOOLS
                _DEFAULT_TOOLS.update(tools)
            else:
                # Initialize _DEFAULT_TOOLS as tools
                _DEFAULT_TOOLS = tools
        else:
            raise ValueError("tools must be either a list or a dictionary")

    if package is not None:
        _PACKAGE = package
    return _DEFAULT_TOOLS


def asmarkdown(
    text: Optional[str],
    replaceThis: Optional[str] = None,
    withThis: Optional[str] = None,
) -> Optional[str]:
    if text is None:
        return None

    extension_configs = {
        "markdown_link_attr_modifier": {
            "new_tab": "on",
            "no_referrer": "external_only",
            "auto_title": "on",
        }
    }

    result = markdown.markdown(
        text,
        extensions=["tables", "markdown_link_attr_modifier"],
        extension_configs=extension_configs,
    )

    if replaceThis is not None and withThis is not None:
        result = result.replace(replaceThis, withThis)

    return result


def createAssistant(
    name: str,
    instructions: str,
    model: Optional[str],
    tools: Optional[list] = None,
) -> Any:
    """Create an OpenAI Assistant programmatically.

    Args:
        name: Name of the assistant
        instructions: Instructions for the assistant
        model: The model to use
        tools: Array of tools to use, defaults to _DEFAULT_TOOLS if None

    Returns:
        Any: The created assistant object with unique id
    """
    if tools is None:
        tools = list(_DEFAULT_TOOLS.keys())
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    assistant = client.beta.assistants.create(
        name=name, instructions=instructions, tools=get_tools(tools), model=model
    )
    return assistant


def getAssistant(**kwargs) -> Union[Any, list[Any]]:
    """Get an OpenAI Assistant object to run a task with.

    Args:
        **kwargs: Keyword arguments for assistant identification
            name/Name/assistantName: Name of the assistant to retrieve
            id/assistant_id: ID of the assistant to retrieve

    Returns:
        Union[Any, list[Any]]: Single assistant object or list of all assistants

    Note:
        Typically you would not need this function since you would use assistantTask()
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    if not kwargs:
        return client.beta.assistants.list(limit=100)

    assistant = None
    for key, value in kwargs.items():
        if key.lower() in ["name", "assistantname"]:
            assistants = client.beta.assistants.list(limit=100)
            for a in assistants.data:
                if a.name == value:
                    assistant = a
                    break
            break
        elif key in ["assistant_id", "id"]:
            assistant = client.beta.assistants.retrieve(assistant_id=value)
            break
    return assistant


def _getf(functionName: str) -> Optional[Callable]:
    """Get a callable function from a string in the form module:function."""
    try:
        if ":" not in functionName:
            found = _DEFAULT_TOOLS[functionName]
            if found is None:
                raise ValueError(
                    f"Function name {functionName} not found in _DEFAULT_TOOLS"
                )
            module_name = found["module"]
            function = functionName
        else:
            module_name, function = functionName.split(":")

        if "." not in module_name:
            module_name = f"{_PACKAGE}.{module_name}"

        module = importlib.import_module(module_name)
        f = getattr(module, function, None)

        if f is None:
            raise AttributeError(
                f"Function {function} not found in module {module_name}"
            )

        return f
    except Exception:
        if settings.DEBUG:
            raise
        return None


def get_tools(array: list[str], value: Optional[str] = None) -> list[dict]:
    """Convert tool strings to callable function dictionaries.

    Args:
        array: List of tool strings in the format "module:function"
        value: Optional function name to filter for

    Returns:
        list[dict]: List of tool dictionaries with type and function
    """
    tools = []
    for a in array:
        f = _getf(a)
        if f is not None:
            if value is not None and a.endswith(":" + value):
                return [{"type": "function", "function": f}]
            tools.append({"type": "function", "function": f})
    return tools


def call_tools_delay(
    combo_id: str, tool_calls: list, tools: Optional[list] = None
) -> Optional[Any]:
    """Process tool calls from the Assistant API and create a Celery chord.

    Args:
        combo_id: String with runId and threadId separated by comma
        tool_calls: Array of tool calls from OpenAI
        tools: Optional array of tools to match with tool_calls

    Returns:
        Optional[Any]: Celery chord group if tasks exist, None otherwise
    """
    tasks = []
    gr = None

    if tools is not None:
        set_default_tools(tools=tools)

    for t in tool_calls:
        #functionCall = getTools(tools,t.function.name)
        tasks.append( _callTool.s( {"tool_call_id": t.id,"function": t.function.name, "arguments" :t.function.arguments }, comboId=comboId) )
        print('function call added to chain '+t.function.name+'('+t.function.arguments+')')

    if len(tasks) > 0:
        # Create a group for function calls and submission
        gr = chord(tasks, submit_tool_outputs.s(combo_id))
        gr.delay()
    return gr


@shared_task(name="call single tool")
def _callTool(tool_call: dict, comboId: Optional[str] = None) -> dict:
    """
    Call a single tool as a Celery task.

    Args:
        tool_call: Dictionary containing function name and arguments
        comboId: Optional thread/run combo ID string

    Returns:
        Dictionary containing comboId and tool output
    """
    functionName = tool_call["function"]
    attributes = json.loads(tool_call["arguments"])

@shared_task(name="call single tool")
def _callTool(tool_call: dict, comboId: Optional[str] = None) -> dict:
    """
    Call a single tool as a Celery task.

    Args:
        tool_call: Dictionary containing function name and arguments
        comboId: Optional thread/run combo ID string

    Returns:
        Dictionary containing comboId and tool output
    """
    functionName = tool_call["function"]
    attributes = json.loads(tool_call["arguments"])

    try:
        call = _getf(functionName)
        if call is None:
            raise ValueError(f"Function {functionName} not found")

        parameter_class = None
        signature = inspect.signature(call)
        parameter = next(iter(signature.parameters.values()), None)

        if (
            parameter
            and parameter.annotation
            and parameter.name == "params"
            and inspect.isclass(parameter.annotation)
            and issubclass(parameter.annotation, BaseModel)
        ):
            parameter_class = parameter.annotation
            attributes = parameter_class(**attributes)
        else:
            attributes["comboId"] = comboId

        functionResponse = call(attributes)
    except Exception as exc:
        if settings.DEBUG:
            # in debug we really like to see what went wrong
            raise Exception(
                f"Error calling {functionName}: {tool_call['arguments']} - {str(exc)}"
            )
        functionResponse = {
            "status": f"Error in function call {functionName}({tool_call['arguments']})"
        }
        # Submit error response to OpenAI for assistant to handle
    tool_output = {
        "tool_call_id": tool_call["tool_call_id"],
        "output": json.dumps(functionResponse, default=str),
    }
    return {"comboId": comboId, "output": tool_output}


@shared_task(name="Submit Tool Outputs")
def submit_tool_outputs(from_tools: list[dict], combo_id: str) -> str:
    """Submit tool outputs to OpenAI API and schedule status check.

    Args:
        from_tools: List of tool outputs from function calls
        combo_id: Combined run_id and thread_id string

    Returns:
        str: The combo_id string for chaining tasks
    """
    run_id = combo_id.split(",")[0]
    thread_id = combo_id.split(",")[1]
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    print(f"Submit tool outputs {run_id}")
    output = []
    for tool in from_tools:
        output.append(tool["output"])
    try:
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id, run_id=run_id, tool_outputs=output
        )
    except Exception as exc:
        print(f"submit tool outputs failed: {exc}")
        return combo_id

    get_status.delay(combo_id)
    return combo_id


@shared_task(
    name="Get Run Status",
    retry_policy={"max_retries": 500, "interval_start": 2},
    rate_limit="15/m",
)
def get_status(combo_id: str) -> Optional[str]:
    """Get OpenAI run status and handle completion or actions.

    Args:
        combo_id: Combined run_id and thread_id string for task identification

    Returns:
        Optional[str]: Task response if completed, None otherwise
    """
    run_id = combo_id.split(",")[0]
    thread_id = combo_id.split(",")[1]
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

    if run.status == "completed" or run.status == "requires_action":
        task = OpenaiTask.objects.get(runId=run_id)
        if task is None:
            raise ValueError(f"Run {run_id} not found in task table")

    if run.status == "completed":
        # Run completed - get response and handle completion
        # Schedule completion call if provided
        messages = client.beta.threads.messages.list(thread_id=task.threadId)
        task.response = ""
        for t in messages.data[0].content:
            if t.type == "text":
                task.response += t.text.value.strip()
            else:
                task.response += f"file [{t.image_file.file_id}]"
        task.status = run.status
        task.completed_at = timezone.now()
        task.save()
        if task.completionCall is not None:
            module_name, function_name = task.completionCall.split(":")
            print(f"Running completion call {task.completionCall}")
            module = importlib.import_module(module_name, package=__package__)
            function = getattr(module, function_name)
            function.delay(run_id)
    elif run.status == "requires_action":
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tools = None if task.tools is None else task.tools.split(",")
        call_tools_delay(combo_id, tool_calls, tools)
        task.status = run.status
        task.save()
        # No retry - tools will restart get_status
    elif run.status in [
        "expired",
        "failed",
        "cancelling",
        "cancelled",
    ]:
        task = OpenaiTask.objects.get(runId=run_id)
        task.status = run.status
        task.save()
    else:
        print(f"get status retry {run_id} {run.status}")
        get_status.retry(countdown=2, max_retries=500)
    print(f"Get status {run_id} {task.status}")
    return task.response


class assistantTask:
    """Manage OpenAI assistant tasks with file handling and tool integration.

    This class provides a high-level interface for creating and managing OpenAI
    assistant tasks, including file uploads, tool integration, and response handling.

    Example:
        Create and run a new task:
            task = assistantTask(
                assistant_name='DemoAssistant',
                prompt='Summarize this: https://valor.vc',
                completion_call='demo:printResult',
                metadata={'id': 'vic123'}
            )
            task.createRun()

        Retrieve an existing task:
            task = assistantTask(run_id=task_id)
    """

    @property
    def prompt(self) -> Optional[str]:
        return self._startPrompt

    @property
    def task_id(self) -> Optional[str]:
        return getattr(self.task, "id", None)

    @property
    def assistant_id(self) -> Optional[str]:
        return getattr(self.assistant_object, "id", None)

    @property
    def thread_id(self) -> Optional[str]:
        return getattr(self.threadObject, "id", None)

    @property
    def run_id(self) -> Optional[str]:
        return getattr(self.task, "runId", None)

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            return {}
        else:
            return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @prompt.setter
    def prompt(self, message: str):
        att = []
        visions = []
        for file in self._fileids:
            if file["vision"]:
                visions.append(file["id"])
                continue
            elif file["retrieval"]:
                tools = [{"type": "file_search"}]
            else:
                tools = [{"type": "code_interpreter"}]
            att.append({"file_id": file["id"], "tools": tools})
        self._startPrompt = message
        self.threadObject = self.client.beta.threads.create(messages=[])
        self._message_id = self.client.beta.threads.messages.create(
            thread_id=self.thread_id, content=message, attachments=att, role="user"
        )
        for v in visions:
            self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                content=[
                    {
                        "type": "image_file",
                        "image_file": {"file_id": v, "detail": "high"},
                    }
                ],
                role="user",
            )

    @property
    def tools(self) -> Optional[list[str]]:
        """Get the list of tool identifiers configured for this task."""
        return self._tools

    @tools.setter
    def tools(self, incoming_tools: list):
        """
        Setter for tools.
        - Ensures incoming_tools is a list.
        - Adds new tools to _DEFAULT_TOOLS via set_default_tools().
        - Stores the original incoming_tools list in self._tools.
        """
        if not isinstance(incoming_tools, list):
            raise ValueError(
                "tools must be a list of tool identifiers (e.g., 'module:function')"
            )

        # Update _DEFAULT_TOOLS with incoming tools
        set_default_tools(tools=incoming_tools)

        # Store the original incoming tools
        self._tools = incoming_tools

    def __init__(self, **kwargs):
        """Create an assistant task.

        Args:
            **kwargs: Keyword arguments for task initialization
                assistant_name: Name of the assistant to use
                run_id/thread_id: ID of existing task to retrieve
                prompt: Initial prompt for the assistant
                completion_call: Function to call on completion
                metadata: Additional task metadata
                tools: List of tool identifiers

        The class operates in two modes:
        1. Create new task: Provide assistant_name and optional parameters
        2. Retrieve existing task: Provide run_id or thread_id
        """
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._fileids = []
        self._startPrompt = None
        self._message_id = None
        self.threadObject = None
        self.runObject = None
        self.messages = None
        self.status = None
        self.response = None
        self.error = None
        self.completionCall = None
        self._metadata = None
        self._tools = None
        for key, value in kwargs.items():
            if (
                key == "run_id"
                or key == "runId"
                or key == "comboId"
                or key == "combo_id"
                or key == "thread_id"
                or key == "threadId"
            ):
                if key == "comboId" or key == "combo_id":
                    value = value.split(",")[0]
                if key == "threadId" or key == "thread_id":
                    self.task = OpenaiTask.objects.get(threadId=value)
                else:
                    self.task = OpenaiTask.objects.get(runId=value)
                if self.task is not None:
                    self.runObject = self.client.beta.threads.runs.retrieve(
                        run_id=self.task.runId, thread_id=self.task.threadId
                    )
                    if self.runObject is None:
                        raise ValueError("Run not found")

                    self.status = self.runObject.status
                    self.response = self.task.response
                    self.threadObject = self.client.beta.threads.retrieve(
                        thread_id=self.task.threadId
                    )
                    self._metadata = self.task.meta
            elif key == "assistant_name" or key == "name" or key == "assistantName":
                self._assistant_name = value
                self.assistant_object = getAssistant(name=self._assistant_name)
                if self.assistant_object is None:
                    msg = f"Assistant {self._assistant_name} not found in OpenAI"
                    raise ValueError(msg)
            elif key == "prompt":
                self.prompt = value
            elif key == "completion_call" or key == "completionCall":
                self.completionCall = value
            elif key == "metadata":
                self._metadata = value
            elif key == "tools":
                self.tools = value

    def createRun(self, temperature: Optional[float] = None) -> Optional[str]:
        """Start a new assistant run with the current thread and configuration.

        Creates a new run using the configured assistant and thread, then starts
        monitoring its status. The temperature parameter controls response
        randomness.

        Args:
            temperature: Sampling temperature (0.0-2.0). Higher values increase
                randomness, lower values make output more deterministic.

        Returns:
            Optional[str]: The run ID if successfully created, None if creation
                fails

        Raises:
            ValueError: If no thread_id has been set for this task
        """
        if self.thread_id is None:
            raise ValueError("Thread ID is required")

        print(f"Creating run for thread {self.thread_id}")

        try:
            run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id= self.assistant_id,
            temperature=temperature,
            )
        except Exception as exc:
            print(f"Create thread failed: {exc}")
            return None
        try: 
            self.task = OpenaiTask.objects.create( assistantId=self.assistant_id, runId=run.id, threadId=self.thread_id, completionCall=self.completionCall, tools=None if self.tools is None else ",".join(self.tools), meta = self._metadata    )
        except Exception as e:
            print('create run failed '+str(e))
            return None

        get_status.delay(f"{self.task.runId},{self.task.threadId}")
        return run_obj.id

    def getJsonResponse(self) -> Optional[dict[str, Any]]:
        """Parse the assistant's response as JSON data.

        Automatically strips any 'json' prefix and markdown code block markers
        from the response before attempting to parse.

        Returns:
            Optional[dict[str, Any]]: Parsed JSON data if successful, None if
                parsing fails or no response exists
        """
        if self.response is not None:
            res = self.response.replace("json", "").replace("```", "")

        get_status.delay(f"{self.task.runId},{self.task.threadId}")
        return run_obj.id

    def getJsonResponse(self) -> Optional[dict[str, Any]]:
        """Parse the assistant's response as JSON data.

        Automatically strips any 'json' prefix and markdown code block markers
        from the response before attempting to parse.

        Returns:
            Optional[dict[str, Any]]: Parsed JSON data if successful, None if
                parsing fails or no response exists
        """
        if self.response is not None:
            res = self.response.replace("json", "").replace("```", "")
            try:
                return json.loads(res, strict=False)
            except json.JSONDecodeError:
                return None
        return None

    def getMarkdownResponse(
        self, replace_this: Optional[str] = None, with_this: Optional[str] = None
    ) -> Optional[str]:
        """Format the assistant's response as markdown with optional replacement.

        Args:
            replace_this: Text to find in the response
            with_this: Text to use as replacement

        Returns:
            Optional[str]: Markdown formatted response or None
        """
        return asmarkdown(self.response, replace_this, with_this)

    def uploadFile(
        self,
        file: Optional[Union[str, bytes, BinaryIO]] = None,
        file_content: Optional[bytes] = None,
        filename: str = "",
        vision: bool = False,
        retrieval: bool = False,
        **kwargs,
    ) -> str:
        """Upload a file to OpenAI for use in the Thread.

        Args:
            file: File path, bytes, or file-like object to upload
            file_content: Raw bytes content of the file if not using file parameter
            filename: Name of the file to use
            vision: Whether to use the file for vision tasks
            retrieval: Whether to use the file for retrieval tasks
            **kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            str: The ID of the uploaded file from OpenAI

        Raises:
            ValueError: If filename missing or no file/content provided
        """
        if not filename:
            raise ValueError("filename is required")

        if file_content is None:
            if file is None:
                raise ValueError("Either file or file_content must be provided")
            try:
                if isinstance(file, str):
                    with open(file, "rb") as f:
                        file_content = f.read()
                elif isinstance(file, (bytes, bytearray, memoryview)):
                    file_content = bytes(file)
                elif hasattr(file, "read"):
                    try:
                        file_content = file.read()
                        if not isinstance(file_content, bytes):
                            file_content = bytes(file_content)
                    except (TypeError, AttributeError):
                        raise ValueError("File object must support reading bytes")
                else:
                    raise ValueError(
                        "File must be a path string, bytes, or file-like object"
                    )
            except (AttributeError, IOError) as exc:
                raise ValueError(f"Could not read from file: {exc}")

        file_extension = filename.split(".")[-1].lower()

        image_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]
        vision = vision or file_extension in image_extensions

        return None

    def getMarkdownResponse(
        self, replace_this: Optional[str] = None, with_this: Optional[str] = None
    ) -> Optional[str]:
        """Format the assistant's response as markdown with optional replacement.

        Args:
            replace_this: Text to find in the response
            with_this: Text to use as replacement

        Returns:
            Optional[str]: Markdown formatted response or None
        """
        return asmarkdown(self.response, replace_this, with_this)

    def uploadFile(
        self,
        file: Optional[Union[str, bytes, BinaryIO]] = None,
        file_content: Optional[bytes] = None,
        filename: str = "",
        vision: bool = False,
        retrieval: bool = False,
        **kwargs,
    ) -> str:
        """Upload a file to OpenAI for use in the Thread.

        Args:
            file: File path, bytes, or file-like object to upload
            file_content: Raw bytes content of the file if not using file parameter
            filename: Name of the file to use
            vision: Whether to use the file for vision tasks
            retrieval: Whether to use the file for retrieval tasks
            **kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            str: The ID of the uploaded file from OpenAI

        Raises:
            ValueError: If filename missing or no file/content provided
        """
        if not filename:
            raise ValueError("filename is required")

        if file_content is None:
            if file is None:
                raise ValueError("Either file or file_content must be provided")
            try:
                if isinstance(file, str):
                    with open(file, "rb") as f:
                        file_content = f.read()
                elif isinstance(file, (bytes, bytearray, memoryview)):
                    file_content = bytes(file)
                elif hasattr(file, "read"):
                    try:
                        file_content = file.read()
                        if not isinstance(file_content, bytes):
                            file_content = bytes(file_content)
                    except (TypeError, AttributeError):
                        raise ValueError("File object must support reading bytes")
                else:
                    raise ValueError(
                        "File must be a path string, bytes, or file-like object"
                    )
            except (AttributeError, IOError) as exc:
                raise ValueError(f"Could not read from file: {exc}")

        file_extension = filename.split(".")[-1].lower()

        image_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]
        vision = vision or file_extension in image_extensions

        retrieval_extensions = [
            "c",
            "cs",
            "cpp",
            "doc",
            "docx",
            "html",
            "java",
            "json",
            "md",
            "pdf",
            "php",
            "pptx",
            "py",
            "rb",
            "tex",
            "txt",
            "css",
            "js",
            "sh",
            "ts",
        ]
        retrieval = retrieval or file_extension in retrieval_extensions

        uploadFile = self.client.files.create(
            file=(filename, file_content), purpose="vision" if vision else "assistants"
        )

        self._fileids.append(
            {
                "id": uploadFile.id,
                "filename": filename,
                "vision": vision,
                "retrieval": retrieval,
            }
        )

        return uploadFile.id

    def getLastResponse(self) -> Optional[dict[str, Any]]:
        """Get the last response message from the assistant.

        Returns:
            Optional[dict[str, Any]]: Most recent message with role and content,
                or None if no messages exist

        Raises:
            ValueError: If thread_id is not set
        """
        if not self.thread_id:
            raise ValueError("Thread ID is required")
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
        return messages.data[0] if messages.data else None

    def getAllMessages(self) -> list[dict[str, Any]]:
        """Retrieve all messages from the current thread in chronological order.

        Returns:
            list[dict[str, Any]]: List of message objects containing role,
                content, and associated metadata

        Raises:
            ValueError: If no thread_id has been set for this task
        """
        if not self.thread_id:
            raise ValueError("Thread ID is required")
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
        return messages.data

    def getFullResponse(self) -> str:
        """Get combined text responses from the assistant.

        Returns:
            str: Combined text content from messages
        """
        if not self.thread_id:
            raise ValueError("Thread ID is required")
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
        res = ""
        for m in reversed(messages.data):
            if m.role == "assistant":
                for t in m.content:
                    if t.type == "text":
                        res += t.text.value
        return res

    def retrieveFile(self, file_id: str) -> bytes:
        """Download a file's content from OpenAI's servers.

        Args:
            file_id: OpenAI file identifier to retrieve

        Returns:
            bytes: Raw binary content of the downloaded file

        Raises:
            openai.NotFoundError: If the specified file_id does not exist
            openai.AuthenticationError: If API authentication fails
        """
        return self.client.files.content(file_id=file_id)
