from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
    
class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"Image uploaded by {self.user} on {self.uploaded_at}"
    
    
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    ocr_usage_count = models.IntegerField(default=0)

    def __str__(self):
        return f"Profile of {self.user.username}"

class OCRHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='ocr_history/')
    extracted_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()