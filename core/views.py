from django.shortcuts import render
import base64
from django.core.files.base import ContentFile
# Create your views here.
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import torch
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(self.relu(self.conv2(x)))  # 14 -> 7
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model1 = keras.models.load_model("ysa1.keras")
model2 = keras.models.load_model("ysa2.keras")
model3 = CNN()
model3.load_state_dict(torch.load('mnist_cnn.pth',  map_location='cpu'))
model3.eval()



def index(request):


    if request.method == 'POST':
        image_data = request.POST.get('image_data')

        format, imgstr = image_data.split(';base64,')  # ← Burada doğru girinti olmalı
        ext = format.split('/')[-1]

        image_file = ContentFile(base64.b64decode(imgstr), name='drawing.' + ext)

        img = Image.open(image_file).convert('L')

        # ✅ CNN için: torchvision işlemleri
        transform_cnn = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_tensor_cnn = transform_cnn(img).unsqueeze(0)  # shape: (1, 1, 28, 28)

        # ✅ CNN Tahmin
        with torch.no_grad():
            output = model3(img_tensor_cnn)
            pred_cnn = output.argmax(dim=1).item()

        # ✅ YSA için: Numpy ile normalizasyon
        img_resized = img.resize((28, 28))
        img_array = np.array(img_resized) / 255.0

        # Model 1 için 2D giriş (28x28), Model 2 için 1D (flattened)
        pred_ysa1 = np.argmax(model1.predict(img_array.reshape(1, 28, 28)))
        pred_ysa2 = np.argmax(model2.predict(img_array.reshape(1, 28 * 28)))

        context = {
            'pred_cnn': pred_cnn,
            'pred_ysa1': pred_ysa1,
            'pred_ysa2': pred_ysa2,
        }

        return render(request, 'index.html', context)


    return render(request,'index.html')