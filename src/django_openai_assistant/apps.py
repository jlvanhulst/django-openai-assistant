from django.apps import AppConfig

__all__ = ['AssistantConfig']

class AssistantConfig(AppConfig):
    name = 'django_openai_assistant'
    label = 'django_openai_assistant'
    verbose_name = 'Django OpenAI Assistant'
    default_auto_field = 'django.db.models.AutoField'
