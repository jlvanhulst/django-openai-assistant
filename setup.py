from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(
    name='django-openai-assistant',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.28',
    license='MIT',
    author="Jean-Luc Vanhulst",
    author_email='jl@valor.vc',
    packages=['django_openai_assistant'],
    package_dir={'': 'src'},
    package_data={ 'django_openai_assistant': ['src/django_openai_assistant/*','src/django_openai_assistant/migrations/*']},
    url='https://github.com/jlvanhulst/django-openai',
    keywords='django celery openai assistants',
    install_requires=[
         'openai','markdown'
      ],

)