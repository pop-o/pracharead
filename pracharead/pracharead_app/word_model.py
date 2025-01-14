import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Model Definition
class CRNN(nn.Module):
    def __init__(self, img_size, num_chars, dropout_rate=0.5):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(dropout_rate)

        # Input dimension for LSTM
        self.rnn_input_dim = img_size[0] // 8 * 128

        # First LSTM layer (bidirectional)
        self.lstm1 = nn.LSTM(self.rnn_input_dim, 256, bidirectional=True, batch_first=True)

        # Second LSTM layer (bidirectional)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)

        self.dropout_rnn = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_chars)

    def forward(self, x):
        # Convolutional layers
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        # Prepare the input for LSTM
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # Change shape to (batch_size, width, channels, height)
        x = x.view(b, w, -1)  # Reshape to (batch_size, width, rnn_input_dim)

        # Pass through the first LSTM layer
        x, _ = self.lstm1(x)

        # Pass through the second LSTM layer
        x, _ = self.lstm2(x)

        x = self.dropout_rnn(x)

        # Fully connected layer
        x = self.fc(x)
        return x

# Load character dictionary
def load_char_dict(char_file):
    with open(char_file, "r", encoding='utf-8') as f:
        char_dict = {idx: char for idx, char in enumerate(f.readline().strip())}
        char_dict[len(char_dict)] = ' '  # Add space for blank token
    return char_dict

# Preprocess input image
def preprocess_image(img_path, img_size, is_gray=True):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_size[1], img_size[0]))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return torch.tensor(img, dtype=torch.float32)

# Decode predictions
def decode_predictions(preds, char_dict):
    decoded_texts = []
    for pred in preds:
        text = ""
        prev_char = None
        for idx in pred:
            char = char_dict.get(idx, "")
            if char != prev_char:  # Skip repeated characters
                if char != ' ':   # Exclude the blank token
                    text += char
            prev_char = char
        decoded_texts.append(text)
    return decoded_texts

# Inference function
def infer(image_path, model_path, char_file, img_size=(64, 256), is_gray=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load character dictionary
    char_dict = load_char_dict(char_file)
    
    # Load model
    num_chars = len(char_dict)
    model = CRNN(img_size, num_chars)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess image
    img = preprocess_image(image_path, img_size, is_gray).to(device)

    # Predict
    with torch.no_grad():
        preds = model(img)
        preds = preds.softmax(2).argmax(2)  # Convert to probabilities and get max indices
        preds = preds.squeeze(0).cpu().numpy().tolist()

    # Decode predictions
    decoded_text = decode_predictions([preds], char_dict)[0]
    return decoded_text

