# A Django / Celery scalable OpenAI Assistant Runner.

Assumption: you have Django up and running with Celery. 
(Tested on Redis)

1. pip install django_openai_assistant

2. in settings.py
Add 'django_openai_assistant' to your INSTALLED_APPS [] array  
make sure to have OPENAI_API_KEY defined in settings with your OpenAI key  
create and apply migrations for django_openai_assistant 

3. Create a simple Assistant in https://platform.openai.com/assistants
To begin you probably want one with no functions.
Let's say you called it 'Test Assistant'

4. Use the assistant in your code:

demo.py
```
ffrom django_openai_assistant.assistant import assistantTask
from celery import shared_task

 # Define OPENAI_API_KEY in your settings.py file
 # Add 'django_openai_assistant' to your INSTALLED_APPS in settings.py
 # run python manage.py makemigrations django_openai_assistant and then python managey.pymigrate
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