SECRET_KEY = "test-key"
INSTALLED_APPS = [
    "django_openai_assistant",
]
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}
OPENAI_API_KEY = "test-key"
USE_TZ = False
