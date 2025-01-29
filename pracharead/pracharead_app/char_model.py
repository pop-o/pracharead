import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
from django.conf import settings
sys.stdout.reconfigure(encoding='utf-8')
# CNN Model Definition
class CNNModel(nn.Module):
    def __init__(self, num_classes=63):
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.batch_norm = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 1 * 1, self.num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel(num_classes=63)
model_path = os.path.join(settings.STATICFILES_DIRS[0], 'file/char_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)
# Transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Mapping dictionary
pracha={
    0:['0x11450','guli'],
    1:['0x11451','chi'],
    2:['0x11452','nasi'],
    3:['0x11453','swa'],
    4:['0x11454','pi'],
    5:['0x11455','njaa'],
    6:['0x11456','khu'],
    7:['0x11457','nhasa'],
    8:['0x11458','cyaa'],
    9:['0x11459','gu'],
    10:['0x11400','A'],
    11:['0x11401','AA'],
    12:['0x11402','I'],
    13:['0x11403','II'],
    14:['0x11404','U'],
    15:['0x11405','UU'],
    16:['0x11406','R'],
    17:['0x11407','RR'],
    18:['0x11408','L'],
    19:['0x11409','LL'],
    20:['0x1140A','E'],
    21:['0x1140B','AI'],
    22:['0x1140C','O'],
    23:['0x1140D','AU'],
    24:['0x1140E','KA'],
    25:['0x1140F','KHA'],
    26:['0x11410','GA'],
    27:['0x11411','GHA'],
    28:['0x11412','NGA'],
    29:['0x11413','NGHA'],
    30:['0x11414','CA'],
    31:['0x11415','CHA'],
    32:['0x11416','JA'],
    33:['0x11417','JHA'],
    34:['0x11418','NYA'],
    35:['0x11419','NHYA'],
    36:['0x1141A','TTA'],
    37:['0x1141B','TTHA'],
    38:['0x1141C','DDA'],
    39:['0x1141D','DDHA'],
    40:['0x1141E','NNA'],
    41:['0x1141F','TA'],
    42:['0x11420','THA'],
    43:['0x11421','DA'],
    44:['0x11422','DHA'],
    45:['0x11423','NA'],
    46:['0x11424','NHA'],
    47:['0x11425','PA'],
    48:['0x11426','PHA'],
    49:['0x11427','BA'],
    50:['0x11428','BHA'],
    51:['0x11429','MA'],
    52:['0x1142A','MHA'],
    53:['0x1142B','YA'],
    54:['0x1142C','RA'],
    55:['0x1142D','RHA'],
    56:['0x1142E','LA'],
    57:['0x1142F','LHA'],
    58:['0x11430','WA'],
    59:['0x11431','SHA'],
    60:['0x11432','SSA'],
    61:['0x11433','SA'],
    62:['0x11434','HA'],
}

def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        threshold = 0.5
        if confidence < threshold:
            return None
        return predicted_class
def match_char(output_class):
    if output_class is not None:
        unicode_value = pracha[output_class][0]
        character = pracha[output_class][1]
        return unicode_value, character, output_class
    return None, "No character", None
