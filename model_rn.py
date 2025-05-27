import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


# Трансформации для тестирования
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Загружаем модель и веса
device = "cuda" if torch.cuda.is_available() else "cpu"

model_rn = nn.Sequential()

model_rn.add_module('resnet', models.resnet152(pretrained=False))

# добавим новые слои для классификации для нашей конкретной задачи
model_rn.add_module('relu_1', nn.ReLU())
model_rn.add_module('fc_1', nn.Linear(1000, 8))

model_rn = model_rn.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_rn.parameters(), lr=0.1)


# Функция для предсказания одного изображения
def predict_image(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
        pred = logits.argmax(dim=1).item()
    return pred
