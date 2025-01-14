from django.urls import path
from . import views
from .views import *

urlpatterns = [
    path('', views.home, name='home'),  # Route for the home page
    path('upload/', views.upload_image, name='upload_image'),  # Route to upload an image for OCR
    path('perform_word_ocr/', perform_word_ocr_view, name='perform_word_ocr'),
    path('perform_char_ocr/', perform_char_ocr_view, name='perform_char_ocr'),
    path('profile/', views.profile, name='profile'),
    
    
]
