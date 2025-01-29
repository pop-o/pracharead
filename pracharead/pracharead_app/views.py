from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from .forms import RegisterForm, LoginForm, ImageUploadForm
from .model_utils import *
from django.conf import settings
from django.http import JsonResponse
import os
from .models import UploadedImage,OCRHistory

def home(request): # Home page view
    return render(request, 'pracharead_app/index.html')

def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = RegisterForm()
    return render(request, 'pracharead_app/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = LoginForm()
    return render(request, 'pracharead_app/login.html', {'form': form})

@login_required
def profile(request):
    return render(request, 'accounts/profile.html', {
        'user': request.user,
    })

def perform_word_ocr_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, image.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        result = perform_word_ocr(image_path)
        extracted_text = result if result != '𑑍' else 'Blank Image'

        # Save only if user is logged in
        if request.user.is_authenticated:
            OCRHistory.objects.create(user=request.user, image=image, extracted_text=extracted_text)

        return JsonResponse({'result': extracted_text})

    return JsonResponse({'error': 'Invalid request'}, status=400)

def perform_char_ocr_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, image.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        result = perform_char_ocr(image_path)

        # Save only if user is logged in
        if request.user.is_authenticated:
            OCRHistory.objects.create(user=request.user, image=image, extracted_text=result)

        return JsonResponse({'result': result})

    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required(login_url='login')
def history(request):
    ocr_history = OCRHistory.objects.filter(user=request.user)[:5]
    return render(request, 'pracharead_app/history.html', {'ocr_history': ocr_history})


def delete_all_history(request):
    if request.method == "POST":
        OCRHistory.objects.all().delete()
        return JsonResponse({"success": True})
    return JsonResponse({"error": "Invalid request"}, status=400)