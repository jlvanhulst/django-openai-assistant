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

OPENAI_API_KEY = environ.get("OPENAI_API_KEY")

USE_TZ = True
TIME_ZONE = "UTC"
