from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Route for the home page
    path('register/', views.register, name='register'),  # Route for registration
    path('login/', views.login_view, name='login'),  # Route for login
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),  # Route for logout
    path('perform_word_ocr/', views.perform_word_ocr_view, name='perform_word_ocr'),
    path('perform_char_ocr/', views.perform_char_ocr_view, name='perform_char_ocr'),
    path('history/', views.history, name='history'),
    path('delete-all-history/', views.delete_all_history, name='delete_all_history'),
    path('profile/', views.profile, name='profile'),
]
