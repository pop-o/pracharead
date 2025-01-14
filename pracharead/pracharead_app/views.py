from django.shortcuts import render, redirect, get_object_or_404
from .forms import ImageUploadForm
from .model_utils import *
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .models import UploadedImage
import os
from django.conf import settings
from django.http import JsonResponse

def home(request): # Home page view
    return render(request, 'pracharead_app/index.html')

def upload_image(request):
    output = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()

            # Update usage count
            if request.user.is_authenticated:
                profile = request.user.profile
                profile.ocr_usage_count += 1
                profile.save()
            else:
                request.session['ocr_usage_count'] = request.session.get('ocr_usage_count', 0) + 1

            output = perform_word_ocr(uploaded_image.image.path)
    else:
        form = ImageUploadForm()

    return render(request, 'pracharead_app/upload.html', {'form': form, 'output': output})



def perform_word_ocr_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, image.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        
        result = perform_word_ocr(image_path)
        if result=='𑑍':
           return JsonResponse({'result': 'Blank Image'})
        else:
            return JsonResponse({'result': result})
    return JsonResponse({'error': 'Invalid request'}, status=400)


def perform_char_ocr_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, image.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        
        unicode_char = perform_char_ocr(image_path)
        result = unicode_char
        return JsonResponse({'result': result})
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def profile(request):
    return render(request, 'accounts/profile.html', {
        'user': request.user,
    })
    
def ocr_result(request, image_id):
    image_instance = get_object_or_404(UploadedImage, id=image_id)
    return render(request, 'pracharead_app/ocr_result.html', {'ocr_result': image_instance.ocr_result})