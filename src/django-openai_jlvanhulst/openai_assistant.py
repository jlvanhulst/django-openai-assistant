from django.utils import timezone
from django.db import models
from django.contrib import admin
import os
from celery import shared_task ,chain
import importlib
from openai import OpenAI

import json
import markdown

class OpenaiTask(models.Model):
    assistantId = models.CharField(max_length=64)
    runId = models.CharField(primary_key=True,max_length=64)
    threadId = models.CharField(max_length=64)
    status = models.CharField(max_length=64, default='created')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True)
    response = models.TextField(null=True)
    completionCall = models.TextField(null=True)
    
    def __str__(self):
        return f'{self.task}: {self.status}'

admin.site.register(OpenaiTask)

openai.api_key = os.getenv("OPENAI_API_KEY")


def getOpenaiClient():
    return OpenAI( api_key=settings.OPENAI_API_KEY)

def createAssistant(name,instructions,model, tools):
    ''' Create an OpenAI Assistant
    parameters:
        name - name of the assistant
        instructions - instructions for the assistant
        model - the model to use
        tools - an array of tools to use. They way we do it here is that we assume that the name of the function corresponds 
        to the name of the tool. So if you have a function called 'webscrape' it will be added to the tools array
        as a tool called 'webscrape'. When running an Assistant we will call the function 'webscrape' 
     
    returns: the assistant object as created.  Assistant.id is the unique key that was assigned to the assistant.
    '''
    client = getOpenaiClient()
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=getTools(tools),
        model=model)
    return assistant
        
def getAssistant(**kwargs) -> object:
    ''' Get an OpenAI Assistant object
    
    parameters:
        Name - the name of the assistant
            or
        id / assistant_id - the id of the assistant
        
        returns: the OpenAI assistant object
    '''
    client = getOpenaiClient()
    for key, value in kwargs.items():
        if key == 'name' or key == 'Name' or key == 'assistantName':
            Name = value
            aa = client.beta.assistants.list()
            assistant = None
            for a in aa.data:
                if a.name == Name:
                    assistant = a
                    break
        elif key == 'assistant_id' or key=="id":
            assistant = client.beta.assistants.retrieve(assistant_id=value)  
            break  
    return assistant

@shared_task(name="Get Run Status", retry_policy={'max_retries': 100, "interval_start":2})
def getStatus(comboId):
    ''' Get the status of an OpenAI run and act on it
    
    parameters:
        comboId - a runId,threadId combo in a string
    '''
    run_id = comboId.split(',')[0]
    thread_id = comboId.split(',')[1]
    # Expects a  runid,threadid combo in a string
    client = getOpenaiClient()
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    if run.status == 'completed' or run.status == 'requires_action':
        task = OpenaiTask.objects.get(runId=run_id)
        if task==None:
            raise Exception('Run '+run_id+' not found in task table')
    
    if run.status == 'completed':
        messagages = client.beta.threads.messages.list( thread_id=task.threadId)
        task.response = messagages.data[0].content[0].text.value.strip()
        task.status = run.status
        task.completed_at = timezone.now()
        task.save()
        if task.completionCall != None:
            module = task.completionCall.split(':')[0]
            function =  task.completionCall.split(':')[1]
            print('completion call '+task.completionCall)
            module = importlib.import_module(module,package="chatbot")
            function = getattr(module, function)
            function.delay(run_id)
    elif run.status == 'requires_action':
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        chain( callToolsDelay.s( run_id,task.threadId,tool_calls ) , submitToolOutputs.s( run_id,task ) ).apply_async()
        task.status = run.status
        task.save()
    else:
        print('get status retry '+run_id+' '+run.status)
        getStatus.retry(countdown=1, max_retries=100)
    print('get status '+run_id+' '+task.status)
    return task.response


@shared_task(name="Submit Tool Outputs")
def submitToolOutputs(fromTools, run_id):    
    client = getOpenaiClient()
    print("Submit tool outputs "+run_id)
    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=fromTools["thread_id"],
        run_id=run_id,
        tool_outputs=fromTools["tool_outputs"]
    )
    return run
    
class assistantTask():
    ''' A class to manage an OpenAI assistant task
    
    Basic use:
    
    task = assistantTask(assistantName='DemoAssistant',
        prompt='Summarize the content on this website: https://valor.vc',
        completionCall='demo:printResult'), metadata={'ie':'vic123'}) )
        
    task.createRun()
        
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
    def metadata(self) -> dict:
        return self.runObject.metadata
    
    @prompt.setter
    def prompt(self, message):
        self._startPrompt = message
        self.threadObject = self.client.beta.threads.create(messages=[] )
        self.client.beta.threads.messages.create(
        thread_id=self.thread_id,
        content=message,
        role="user"
            )
    
    def __init__(self, **kwargs ):
        ''' Create an assistant task
        
        Two modes: a new taks or retrieve an existing (probably completed) task from the database
        
        if run_id is provided it will retrieve the task from the database. If not it will create a new task.
        
        '''
        self.client = getOpenaiClient()
        self._startPrompt = None
        self.threadObject = None
        self.runObject = None
        self.messages = None
        self.status = None
        self.response = None
        self.error = None
        self.completionCall = None
        self._metadata = None
        for key, value in kwargs.items():
            if key == 'run_id':
                self.task= OpenaiTask.objects.get(runId=value)
                if self.task != None:
                    self.runObject = self.client.beta.threads.runs.retrieve(run_id=self.task.runId, thread_id=self.task.threadId)
                    if self.runObject == None:
                        raise Exception('Run not found')
                    else:
                        self.status = self.runObject.status
                        self.response = self.task.response
                        self.threadObject = self.client.beta.threads.retrieve(thread_id=self.task.threadId)
                        self._metadata = self.runObject.metadata
            elif key == 'assistantName':
                self.assistantName = value
                self.assistantObject = getAssistant(name=self.assistantName)
            elif key == 'prompt':
                self.prompt = value  
            elif key == 'completionCall':
                self.completionCall = value    
            elif key == 'metadata':
                self._metadata = value  
                  
      
    def createRun(self) -> str:
        ''' Create an OpenAI run and start checking the status 
        
        This this will persist the task in the database - run id is the primary key. Please note that openAI needs both ThreadId and RunId 
        to retrieve a Run. We handle that in the so that you will only need run id. The primary key in the Taks table is the run id.
        
        '''
        print('create run '+self.thread_id)
        # Excepts an openAiTask object    
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id= self.assistant_id,
            metadata=self._metadata
        )
        self.task = OpenaiTask.objects.create( assistantId=self.assistant_id, runId=run.id, threadId=self.thread_id, completionCall=self.completionCall)
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
                return json.loads(res)
            except:
                return None
        
    def markdownResponse(self):
        ''' returns the response with markdown formatting - convient for rendering in chat like responses
        '''
        if self.response != None:
            extension_configs = {
            'markdown_link_attr_modifier': {
            'new_tab': 'on',
            'no_referrer': 'external_only',
            'auto_title': 'on',
            }}
        return markdown.markdown(self.response,extensions=['tables','markdown_link_attr_modifier'],extension_configs=extension_configs)
