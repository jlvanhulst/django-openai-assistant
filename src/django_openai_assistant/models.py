from django.db import models

class OpenaiTask(models.Model):
    assistantId = models.CharField(max_length=64)
    runId = models.CharField(primary_key=True,max_length=64)
    threadId = models.CharField(max_length=64)
    status = models.CharField(max_length=64, default='created')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True)
    response = models.TextField(null=True)
    completionCall = models.TextField(null=True)
    tools = models.TextField(null=True)
    meta = models.JSONField(null=True)
    
    def __str__(self):
        return f'{self.runId}: {self.status}'
    
