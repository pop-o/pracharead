from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import UploadedImage

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image']

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'placeholder': 'Enter your email'}))
    username = forms.CharField(required=True,widget=forms.TextInput(attrs={'placeholder': 'Enter your username'}))
    password1 = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Enter your password'}))
    password2 = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Confirm your password'}))

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}))
