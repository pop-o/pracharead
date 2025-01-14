import os
from django.conf import settings
from django.templatetags.static import static
from .word_model import infer 
from .char_model import predict_image, match_char

def perform_word_ocr(image_path):
    model_path = os.path.join(settings.STATICFILES_DIRS[0], 'file/word_model.pth')
    char_file = os.path.join(settings.STATICFILES_DIRS[0], 'file/charList.txt')
    img_size = (64, 256)
    is_gray = True
    return infer(image_path, model_path, char_file, img_size, is_gray)

def perform_char_ocr(image_path):
    # Use the predict_image function from char_model.py
    predicted_class = predict_image(image_path)
    if predicted_class is not None:
        unicode_value, character, output_class = match_char(predicted_class)
        unicode_char = chr(int(unicode_value, 16))
        return unicode_char
    return "No Character"