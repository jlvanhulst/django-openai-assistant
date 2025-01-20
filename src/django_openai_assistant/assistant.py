from django.conf import settings
from django.utils import timezone
from django.db import models
from celery import shared_task, chord
from typing import Optional
import json
import markdown
from openai import OpenAI
import importlib
from .models import OpenaiTask
import inspect
from pydantic import BaseModel
_DEFAULT_TOOLS = {}
_PACKAGE = None

def set_default_tools(tools: Optional[list] = None, package: Optional[str] = None):
    '''
    Set the default tools to use for the assistant. This is a global setting and will be used for all assistants 
    if you don't provide a tools array when creating an assistant.

    Parameters:
        tools - a list of strings in the form <module>:<function> where module is the module name and function is the function name
                tools = [ "salesforce:getCompany", "salesforce:getContact", ...]
                
                Optionally, include the package in the module name:
                tools = [ "chatbot.salesforce:getCompany", "chatbot.salesforce:getContact", ...]
        
        package - the package to use for the tools. (if not included in the module name ie not '.' in the module name) 
                  This saves you from having to add the package name for each module.
    
    Behavior:
        - If _DEFAULT_TOOLS is already a dictionary, add new tools to it.
        - If tools is a list, convert it to a dictionary and merge.
        - If tools is None, do nothing.

    Returns:
        The updated _DEFAULT_TOOLS dictionary.
    '''
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

def asmarkdown(text, replaceThis=None, withThis=None):
    ret = None
    if text != None:
        extension_configs = {
        'markdown_link_attr_modifier': {
        'new_tab': 'on',
        'no_referrer': 'external_only',
        'auto_title': 'on',
        }}
        if replaceThis != None and withThis != None:
            ret = markdown.markdown(text,extensions=['tables','markdown_link_attr_modifier'],extension_configs=extension_configs).replace(replaceThis,withThis)
        else:
            ret =markdown.markdown(text,extensions=['tables','markdown_link_attr_modifier'],extension_configs=extension_configs)
    return ret


def createAssistant(name:str,instructions:str,model:Optional[str], tools: Optional[list]=_DEFAULT_TOOLS) -> object:
    ''' Create an OpenAI Assistant programmatically
    parameters:
        name - name of the assistant 
        instructions - instructions for the assistant
        model - the model to use
        tools - an array of tools to use. They way we do it here is that we assume that the name of the function corresponds 
        to the name of the tool. So if you have a function called 'webscrape' it will be added to the tools array
        as a tool called 'webscrape'. When running an Assistant we will call the function 'webscrape' 
     
        Only use to create new Assistants in OpenAI!
        
    returns: the assistant object as created.  Assistant.id is the unique key that was assigned to the assistant.
    '''
    client = OpenAI( api_key=settings.OPENAI_API_KEY)
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=getTools(tools),
        model=model)
    return assistant
        
        
def getAssistant(**kwargs) -> object:
    ''' Get an OpenAI Assistant object to run a task with
    
    Typically you would not need this function,  since you would use assistantTask()
     
    parameters:
        Name - the name of the assistant
            or
        id / assistant_id - the id of the assistant
        
        when called with no parameters it will return a list of all assistants
        
        returns: the OpenAI assistant object
        
    '''
    client = OpenAI( api_key=settings.OPENAI_API_KEY)
    if len(kwargs.items())==0:
        return client.beta.assistants.list(limit=100)

    for key, value in kwargs.items():
        if key == 'name' or key == 'Name' or key == 'assistantName':
            Name = value
            aa = client.beta.assistants.list(limit=100)
            assistant = None
            for a in aa.data:
                if a.name == Name:
                    assistant = a
                    break
        elif key == 'assistant_id' or key=="id":
            assistant = client.beta.assistants.retrieve(assistant_id=value)  
            break  
    return assistant


def _getf(functionName) -> object:
    ''' get a callable function from a string in the form module:function
    '''
    try:
        if not ":" in functionName:
            # use the _DEFAULT_TOOLS array
            found = _DEFAULT_TOOLS[functionName]
            if found == None:
                raise Exception(f'Function name {functionName} not defined in module {module_name}')
            module_name = found["module"]
        else:
            module_name = functionName.split(':')[0]
            function =  functionName.split(':')[1]
        if not "." in module_name:
            module_name = f"{_PACKAGE}.{module_name}"
        module = importlib.import_module(module_name)

    except:
        raise Exception(f'Function name {function} not defined in module {module_name}')

    f = None
    try:
        f = getattr(module, functionName,None)
    except:
        if settings.DEBUG:
            raise Exception('Function '+functionName+' could not run')
        else:
            pass
    return f

def getTools(array,value=None) -> list|object:
    ''' takes an array of tools in the form <modul>:<function> and return an array with the callable functions
    if value is provided will return the callable function for just that function. 
    in that case expecting JUST the function name for example 'getCompany' and not '.salesforce:getCompany'
    
    '''
    tools = []
    for a in array:
        f = _getf(a)
        if f!=None:
            if not value==None and a.endswith(':'+value):
                return a
            tools.append( {"type": "function","function" : f()})       
    return tools


def callToolsDelay(comboId,tool_calls,tools):
    ''' process the tools_calls array we get from the Assistant API
    
    comboId is a string with the runId and threadId separated by a comma
    tool_calls is the array straight from OpenAI 
    tools is our array of tools that we use to match the tool_calls to the functions we need to call    
    
    the function will create celery chord() to execute the calls and trigger the submit function with all calls
    are completed.
    
    '''
    # receives COMBO ID
    tasks = []
    gr = None
    # legacy support to make sure that the tools are in _DEFAULT_TOOLS
    if not tools == None:
        set_default_tools(tools=tools)
        
    for t in tool_calls:
        #functionCall = getTools(tools,t.function.name)
        tasks.append( _callTool.s( {"tool_call_id": t.id,"function": t.function.name, "arguments" :t.function.arguments }, comboId=comboId) )
        print('function call added to chain '+t.function.name+'('+t.function.arguments+')')

    if len(tasks)>0:
        # create a group will all the functions to call and the submit function to be called after those are done
        gr = chord(tasks,submitToolOutputs.s(comboId) )
        gr.delay()
    return gr


@shared_task(name="call single tool") 
def _callTool(tool_call,comboId=None):
    ''' call a single tool. Since this is a shared task, each individual function call 'becomes' a task
    note that the function we're calling is not expected to be a task. It's just a function.
    
    Also not that we're adding the thread/run combo id string as an extra named parameter. This allows you to retrieve the callind task or the run/thread if need.
    '''
    functionName = tool_call['function']
    attributes = json.loads(tool_call['arguments'])
    attributes['comboId'] = comboId
    try:
        parameter_class = None
        call = _getf(functionName) # get the callable function object
        # check if the function argument is params and pydantic schema, in that case instantiate the schema object and pass that
        signature = inspect.signature(call  )
        parameter = next(iter(signature.parameters.values()), None)  # Get the first parameter
        if parameter and parameter.annotation and parameter.name =='params' and inspect.isclass(parameter.annotation) and issubclass(parameter.annotation, BaseModel) :
            parameter_class = parameter.annotation  # e.g. AddLocationToCalendarEvent
        if parameter_class:
            attributes  = parameter_class(**attributes)
            
        functionResponse =call(attributes) # save the response
    except Exception as e:
        if settings.DEBUG:
            # in debug we really like to see what went wrong 
            raise Exception('Error in function call '+functionName+'('+tool_call['arguments']+') '+str(e))
        functionResponse = { "status" : 'Error in function call '+functionName+'('+tool_call['arguments']+')' }
        # we can submit this repsons back to OpenAI so that Assistant will continue and report it encountered a problem
    tool_output =   { "tool_call_id": tool_call['tool_call_id'] , "output": json.dumps(functionResponse,default=str) }
    return {"comboId" : comboId , "output":tool_output }


@shared_task(name="Submit Tool Outputs")
def submitToolOutputs(fromTools, comboId):    
    ''' This function gets called after the last function call is completed and will submit the results back
    to the AssistantsAPI
    
    fromTools is the array of outputs from each function
    
    After submitting we schedule the next status check to see if the run completed. 
    '''
    run_id = comboId.split(',')[0]
    thread_id = comboId.split(',')[1]
    client = OpenAI( api_key=settings.OPENAI_API_KEY)
    print("Submit tool outputs "+run_id)
    output = []
    for o in fromTools:
        output.append( o["output"])
    try:
        run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=output
        )
    except Exception as e:
        print('submit tool outputs failed '+str(e))
        return comboId
    # now schedule the next status check to see if the run is done.
    getStatus.delay(comboId)
    return comboId


@shared_task(name="Get Run Status", retry_policy={'max_retries': 500, "interval_start":2}, rate_limit='15/m')
def getStatus(comboId):
    ''' Get the status of an OpenAI run and act on it
    
    parameters:
        comboId - a runId,threadId combo in a string
        we use this because of the celery parameter passing simplicity
        
    '''
    run_id = comboId.split(',')[0]
    thread_id = comboId.split(',')[1]
    # Expects a  runid,threadid combo in a string
    client = OpenAI( api_key=settings.OPENAI_API_KEY)
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    if run.status == 'completed' or run.status == 'requires_action':
        # only then do we need the object to do something with. 
        task = OpenaiTask.objects.get(runId=run_id)
        if task==None:
            raise Exception('Run '+run_id+' not found in task table')
    
    if run.status == 'completed':
        # Run is done! Get the response, save in the task opbject and schedule the completion call if one was 
        # provided.
        messages = client.beta.threads.messages.list( thread_id=task.threadId)
        task.response =""
        for t in messages.data[0].content:
            if t.type == 'text':
                task.response += t.text.value.strip()
            else:
                task.response += 'file ['+t.image_file.file_id+']'
        task.status = run.status
        task.completed_at = timezone.now()
        task.save()
        if task.completionCall != None:
            module = task.completionCall.split(':')[0]
            function =  task.completionCall.split(':')[1]
            print('completion call '+task.completionCall)
            module = importlib.import_module(module,package=__package__)
            function = getattr(module, function)
            function.delay(run_id)
    elif run.status == 'requires_action':
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tools = task.tools.split(',')
        callToolsDelay(comboId,tool_calls,tools)
        task.status = run.status
        task.save()
        #getStatus.retry(countdown=1, max_retries=100)
        # no retry here - the calltoolsday will run all tools and then restart a getstatus()
    elif run.status == 'expired' or run.status == 'failed' or run.status == 'cancelling' or run.status == 'cancelled' or run.status == 'expired':
        task = OpenaiTask.objects.get(runId=run_id)
        task.status = run.status
        task.save()
    else:  
        print('get status retry '+run_id+' '+run.status)
        getStatus.retry(countdown=2, max_retries=500)
    print('get status '+run_id+' '+task.status)
    return task.response

    
class assistantTask():
    ''' A class to manage an OpenAI assistant task
    
    Basic use:
    
    task = assistantTask(assistantName='DemoAssistant',
        prompt='Summarize the content on this website: https://valor.vc',
        completionCall='demo:printResult'), metadata={'ie':'vic123'}) )    
    task.createRun()  # start the Assistant
    
    retrieve a task:
    
    task = assistantTask(run_id = taskID)

    '''
    @property
    def prompt(self) -> str:
        return self._startPrompt
    
    @property
    def task_id(self) ->str:
        return self._task_id

    @property
    def assistant_id(self) -> str:
        return self.assistantObject.id
    
    @property
    def thread_id(self) -> str:
        return self.threadObject.id
    
    @property
    def run_id(self) -> str:
        return self.task.runId
    
    @property
    def metadata(self) -> dict:
        if self._metadata == None:
            return {}
        else:
            return self._metadata
    
    @metadata.setter
    def metadata(self, value):
        self._metadata = value
        
    @prompt.setter
    def prompt(self, message:str):
        att = []
        visions = []
        for file in self._fileids:
            if file['vision']:
                visions.append(file['id'])
                continue
            elif file['retrieval']:
                tools = [{'type': 'file_search'}]
            else:
                tools = [{'type': 'code_interpreter'}]            
            att.append( {
                'file_id': file['id'],
                'tools': tools  }  ) 
        self._startPrompt = message
        self.threadObject = self.client.beta.threads.create(messages=[] )
        self._message_id = self.client.beta.threads.messages.create(
        thread_id=self.thread_id,
        content=message,
        attachments= att,
        role="user"
            )
        for v in visions:
            self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                content= [{
                'type' : "image_file",
                'image_file' : {"file_id": v ,'detail':'high'}}],
                role="user"
            )
    
    @property
    def tools(self) -> list:
        '''Getter for tools. Returns the original tools list provided during initialization.'''
        return self._tools

    @tools.setter
    def tools(self, incoming_tools: list):
        '''
        Setter for tools.
        - Ensures incoming_tools is a list.
        - Adds new tools to _DEFAULT_TOOLS via set_default_tools().
        - Stores the original incoming_tools list in self._tools.
        '''
        if not isinstance(incoming_tools, list):
            raise ValueError("tools must be a list of tool identifiers (e.g., 'module:function')")

        # Update _DEFAULT_TOOLS with incoming tools
        set_default_tools(tools=incoming_tools)

        # Store the original incoming tools
        self._tools = incoming_tools  
    
    def __init__(self, **kwargs ):
        ''' Create an assistant task
        
        Two modes: create a new task or retrieve an existing (probably completed) task from the database
        
        if run_id is provided it will retrieve the task from the database. If not it will create a new task.
        
        '''
        self.client = OpenAI( api_key=settings.OPENAI_API_KEY)
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
        
        for key, value in kwargs.items():
            if key == 'run_id' or key == 'runId' or key == "comboId" or key =='thread_id' or key == 'threadId':
                if key == 'comboId':
                    value = value.split(',')[0]
                if key == 'threadId' or key == 'thread_id':
                    self.task= OpenaiTask.objects.get(threadId=value)
                else:
                    self.task= OpenaiTask.objects.get(runId=value)
                if self.task != None:
                    self.runObject = self.client.beta.threads.runs.retrieve(run_id=self.task.runId, thread_id=self.task.threadId)
                    if self.runObject == None:
                        raise Exception('Run not found')
                    else:
                        self.status = self.runObject.status
                        self.response = self.task.response
                        self.threadObject = self.client.beta.threads.retrieve(thread_id=self.task.threadId)
                        self._metadata = self.task.meta
            elif key == 'assistantName':
                self.assistantName = value
                self.assistantObject = getAssistant(name=self.assistantName)
                if self.assistantObject == None:
                    raise Exception('Assistant '+self.assistantName+' not found in openAI call')
            elif key == 'prompt':
                self.prompt = value  
            elif key == 'completionCall':
                self.completionCall = value    
            elif key == 'metadata':
                self._metadata = value  
            elif key == 'tools':
                self.tools = value    
     
      
    def createRun(self,temperature:float=1) -> str:
        ''' Create an OpenAI run and start checking the status 
        
        This this will persist the task in the database - run id is the primary key. Please note that openAI needs both ThreadId and RunId 
        to retrieve a Run. We handle that in the object so that you will only need run id. The primary key in the Taks table is the run id.
        
        paramaters:
            temperature - the temperature to use for the run. Default is 1 values 0 - 2 as per OpenAI documentation
        
        '''
        print('create run '+self.thread_id)
        # Excepts an openAiTask object    
        try:
            run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id= self.assistant_id,
            temperature=temperature,
            )
        except Exception as e:
            print('create thread failed '+str(e))
            return None
        try: 
            self.task = OpenaiTask.objects.create( assistantId=self.assistant_id, runId=run.id, threadId=self.thread_id, completionCall=self.completionCall, tools=",".join(self.tools), meta = self._metadata    )
        except Exception as e:
            print('create run failed '+str(e))
            return None
        self.task.save()
        # start the delayed status checking
        getStatus.delay(self.task.runId+','+self.task.threadId)
        return run.id
     
       
    def jsonResponse(self):
        ''' Try to convert the openai response to Json. When we ask openai explicitly for json it will return a string with json in it.
            This function will try to convert that string to json. If it fails it will return None
            OpenAI often insists on added 'json' at the begining and the triple quotes
        '''
        if self.response != None:
            res = self.response.replace('json','').replace("```",'')
            try:
                return json.loads(res,strict=False)
            except:
                return None
        
    def markdownResponse(self, replaceThis=None, withThis=None):
        ''' returns the response with markdown formatting - convient for rendering in chat like responses
        '''
        return asmarkdown(self.response,replaceThis,withThis)
        
        
    def uploadFile(self,file=None,fileContent=None,filename=None,**kwargs):
        ''' Upload a file to openAI either for the Assistant or for the Thread.
        
        parameters:
            file - a file object
            fileContent - the content of the file
            
            addToAssistant - if true will add the file to the assistant. If false will add it to the thread
            *** Note addToAssistant is not currently supported due to the V2 changes. 
            
            filename - the name of the file. If not provided will use the name of the file object
            All uploaded files will automatically be provided in the message to the assistant with both search and code interpreter enabled.
        '''
        if fileContent == None:
            fileContent = file.read()
        # Determine file extension
        file_extension = filename.split('.')[-1].lower()
        
        # Determine if the file is an image
        image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']
        vision = file_extension in image_extensions
        # Determine if the file is for retrieval
        retrieval_extensions = [
            'c', 'cs', 'cpp', 'doc', 'docx', 'html', 'java', 'json', 'md', 'pdf', 'php',
            'pptx', 'py', 'rb', 'tex', 'txt', 'css', 'js', 'sh', 'ts'
        ]
        retrieval = file_extension in retrieval_extensions
        
        uploadFile = self.client.files.create( file=(filename,fileContent),purpose=('vision' if vision else 'assistants'))
        #uploadFile = self.client.files.create( file=(filename,fileContent),purpose='assistants')

        # Append the file information to self._fileids
        self._fileids.append({
            'id': uploadFile.id,
            'filename': filename,
            'vision': vision,
            'retrieval': retrieval
        })
    
        return uploadFile.id
        
    def getlastresponse(self):
        ''' Get the last response from the assistant, returns messages.data[0] 
        '''
        messages = self.client.beta.threads.messages.list( thread_id=self.thread_id)
        return messages.data[0]
    
    def getallmessages(self):
        ''' Get all messages from the assistant - returns messages.data (list)
        '''
        messages = self.client.beta.threads.messages.list( thread_id=self.thread_id)
        return messages.data
    
    def getfullresponse(self):
        ''' Get the full text response from the assistant (concatenated text type messages)
        traverses the messages.data list and concatenates all text messages
        '''
        messages = self.client.beta.threads.messages.list( thread_id=self.thread_id)
        res = ''
        for m in reversed(messages.data):
            if m.role == 'assistant':
                for t in m.content:
                    if t.type == 'text':
                        res += t.text.value
        return res
    
    def retrieveFile(self,file_id):
        ''' Retrieve the FILE CONTENT of a file from OpenAI
        '''
        return self.client.files.content(file_id=file_id)