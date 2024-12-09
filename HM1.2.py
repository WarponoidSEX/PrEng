# import torch
# from torchvision import models, transforms
# from PIL import Image

# # Загрузка предобученной модели
# model = models.resnet50(pretrained=True)  # Использование ResNet50 вместо ResNet18
# model.eval()

# # Преобразование изображения
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.RandomHorizontalFlip(),  # Случайное горизонтальное отражение
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Изменение цвета
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Загрузка изображения
# img = Image.open("image3.jpg")
# img_t = transform(img)
# batch_t = torch.unsqueeze(img_t, 0)

# # Предсказание
# output = model(batch_t)
# _, pred = torch.max(output, 1)

# # Словарь классов ImageNet
# imagenet_classes_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
# import requests
# imagenet_classes = requests.get(imagenet_classes_url).json()

# # Получение текстового названия класса
# predicted_class = imagenet_classes[pred.item()]
# print(f"Predicted class: {predicted_class}")
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Загрузка предобученной модели ViT
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Обработка изображения
image = Image.open("image4.jpg")
inputs = processor(images=image, return_tensors="pt")

# Предсказание
outputs = model(**inputs)
predicted_class_idx = outputs.logits.argmax(-1).item()

# Получение названия класса
predicted_class_label = model.config.id2label[predicted_class_idx]
print(f"Predicted class: {predicted_class_label}")
