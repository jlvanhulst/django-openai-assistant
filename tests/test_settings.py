from os import environ

SECRET_KEY = environ.get("DJANGO_SECRET_KEY", "dummy-secret-key-for-tests")
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django_openai_assistant",
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

OPENAI_API_KEY = environ.get("OPENAI_API_KEY", "dummy-key-for-testing")
REDIS_URL = environ.get("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

USE_TZ = True
TIME_ZONE = "UTC"
