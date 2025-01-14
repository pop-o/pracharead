from django.shortcuts import redirect
from django.conf import settings

class UsageLimitMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            profile = request.user.profile
            if profile.ocr_usage_count >= 10 and 'login' not in request.path:
                return redirect('login')  # Redirect if limit is exceeded
        else:
            usage_count = request.session.get('ocr_usage_count', 0)
            if usage_count >= 10:
                return redirect('login')  # Redirect if anonymous user exceeds limit

        response = self.get_response(request)
        return response
