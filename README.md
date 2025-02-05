# A Django / Celery scalable OpenAI Assistant Runner.

Assumption: you have Django up and running with Celery. 
(Tested on Redis)

1. In your terminal:
```bash
pip install django_openai_assistant
```

2. in `settings.py`
 - Add 'django_openai_assistant' to your INSTALLED_APPS [] array:
```py
INSTALLED_APPS = [
    # other apps
    'django_openai_assistant',
]
```
 - Make sure to have OPENAI_API_KEY defined in settings with your OpenAI key:
```py
OPENAI_API_KEY = "<your-key>"
```
 - Create and apply migrations for django_openai_assistant:
```py
python manage.py makemigrations django_openai_assistant
python manage.py migrate django_openai_assistant
```
3. Create a simple Assistant in https://platform.openai.com/assistants. To begin you probably want one with no functions.

4. In step (3) let's say you called it 'Test Assistant', then use the assistant in your code:

demo.py
```py
from django_openai_assistant.assistant import assistantTask
from celery import shared_task

 # Define OPENAI_API_KEY in your settings.py file
 # Add 'django_openai_assistant' to your INSTALLED_APPS in settings.py
 # run python manage.py makemigrations django_openai_assistant
 # run python manage.py migrate
 # create at least one Assistant in https://platform.openai.com/assistants

def testAssistant(request=None):
    # replace <<your appname>> with the name you django app this file is in!
    # replace <<your assistant name>> with the name of the assistant you created in the OpenAI platform
    task = assistantTask(assistantName="<<your assistant name>>", tools= [], completionCall = "<<your appname>>.test:afterRunFunction")
    task.prompt = "Who was Napoleon Bonaparte?"
    task.createRun() ## this will get everything going!

@shared_task(Name="This will be called once the run is complete")
def afterRunFunction(taskID):
    # Function is called when run is completed. MUST be annoted as @shared_task!!! 
    # start by retrieving the task
    task = assistantTask(run_id = taskID)
    if task.status == 'completed': ## check to make sure it is completed not failed or something else
        print(task.markdownresponse())
    else:
        print('run failed')
```
See https://medium.com/@jlvalorvc/building-a-scalable-openai-assistant-processor-in-django-with-celery-a61a1af722e0


## Version history:
0.7.4
- First version with some Devin 'help'. Working on a testable, lintable version. Stay tuned and watch along in github. 
- Temperature has been made optional (instead of default 1 when not provided) because the new o1 / o3 models don't allow it.

0.7.3
- remove injection of comboid when using class parameters because it will fail (if the class is set to strict)

0.7.2
- bug fix for calling without tools. (regression from 0.7.1)

0.7.1 
- better support for legacy where individual call with tools =[] and those tools are NOT in the set_default_tools() are automatically added.
- please note: the celery WORKER needs to do a call to set_default_tools(). Best place in django read()
 
0.7.0 
- Major update for tool calling
Added set_default_tools() to set the default tools for all assistants. 
This means you can now ommit the tools parameter when creating an assistant call
you can now add a set_default_tools() call in the beginning of your code and it will be used for any agent call.
(Note you still need to have the tools defined in the OpenAI platform for each assistant)

Another big chance is a Pydanctic class is now automatically detected and supported. The first parameter of your tools function should be call params for this. It will be checked to be subclass of BaseModal. If so, the OpenAI parameters will be converted through Pydantic.

Example:


~~~
class CreateEventParams(BaseModel):
    email: str = Field(..., description="Owner's email.")
    start: str = Field(..., description="Event start (ISO 8601).")
    end: str = Field(..., description="Event end (ISO 8601).")
    title: str = Field(..., description="Event title.")
    description: Optional[str] = Field('', description="Event description.")
    attendees: Optional[List[str]] = Field([], description="Attendee emails.")
    address: Optional[str] = Field(None, description="Event address.")
    add_google_meet_link: Optional[bool] = Field(False, description="Add Google Meet link.")
    calendar_id: Optional[str] = Field('primary', description="Calendar ID.")
    time_zone: Optional[str] = Field('America/New_York', description="Time zone identifier.")

# params is a Pydantic (sub)class 
def create_event(params: CreateEventParams) -> Dict:
    '''
    Create an event in the Google calendar 
    '''
    # Validate time_zone
    try:
        ZoneInfo(params.time_zone)
    except Exception:
        raise ValueError(f"Invalid time zone: {params.time_zone}")

    # Convert start/end to datetime
    try:
        start_datetime = datetime.fromisoformat(params.start)
    except ValueError:
        raise ValueError("Invalid ISO 8601 format for 'start'.")

    try:
        end_datetime = datetime.fromisoformat(params.end)
    except ValueError:
        raise ValueError("Invalid ISO 8601 format for 'end'.")

    # Call create_calendar_event function
    event = create_calendar_event(
        email=params.email,
        start=start_datetime,
        end=end_datetime,
        title=params.title,
        description=params.description,
        attendees=params.attendees,
        address=params.address,
        add_google_meet_link=params.add_google_meet_link,
        calendar_id=params.calendar_id,
        time_zone=params.time_zone
    )
    return event.model_dump()
~~~

This way allows to also automatically create a schema for a function call! (coming soon!)

0.6.0
- since 1.33.00 Openai properly supports the 'vision' file attachment. Now supported here as well. 
Based on filetype images are automatically marked as 'vision' and added to the thread as such.
        image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']

- Added getallmessages() and getfullresponse() to easily support threaded responses 
- requires openai 1.33.0

0.5.4
- added getallmessages() which returns threads.messages.list( thread_id=self.thread_id) .data
0.5.3
- Added getfullresponse() that comiles all Assistant repsonses in one message
- still waiting for the openai Python library to support vision.
0.5.2
- Updated the file upload mechanism to determine file types (retrieval yes/no and prepare for vision support (image yes/no))
- When attachments are added to a thread they will always have 'tools' enabled and retrieval will only be enabled for the supported file types https://platform.openai.com/docs/assistants/tools/file-search/supported-files

0.5.1
- Remove getopenaiclient() instead use OpenAI() everywhere
- Fix getAssistant() that could fail if retrieving an Assistant by name from an org with more than 20 Assistants.

0.5.0
- Added support for Assistants 2.0. For now all files are added to a thread with support for both search and code completion. For now no support to upload files to a vectorstore to an Assistant. 

0.4.3
- Added optional parameter temperature createRun() default is 1, like the OpenAI default

0.4.2
- Added optional parameter temperature createRun() default is 1, like the OpenAI default

0.4.1
- another fix for metadata always returning {} never None

0.4.0
- redo readme.md 
- standard version numbering
- task.metadata is now initiatilized as {} to prevent task.metadata.get() from failing.

0.33
- retrieve by thread_id or run_id openaiTask(run_id='run_xxxx').   
- asmarkdown(string) available as a function outside class  
- retrievefile() to download a file by openai file ID.  
- now processing multi response with embedded image.

0.32
- small fixes.  

0.31
- made sure that file uploads to openai receive a 'file name' when uploaded. 

0.30 
- fix to properly differentiate between two functions that start with the same name like 'company'
- and 'companyfind' and throw and exception (istead of a pass) when running in Debug mode when calling tool function. 
