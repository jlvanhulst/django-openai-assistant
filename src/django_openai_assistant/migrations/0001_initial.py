# Generated by Django 4.2.5 on 2023-12-29 23:38

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='OpenaiTask',
            fields=[
                ('assistantId', models.CharField(max_length=64)),
                ('runId', models.CharField(max_length=64, primary_key=True, serialize=False)),
                ('threadId', models.CharField(max_length=64)),
                ('status', models.CharField(default='created', max_length=64)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('completed_at', models.DateTimeField(null=True)),
                ('response', models.TextField(null=True)),
                ('completionCall', models.TextField(null=True)),
                ('tools', models.TextField(null=True)),
                ('meta', models.JSONField(null=True)),
            ],
        ),
    ]