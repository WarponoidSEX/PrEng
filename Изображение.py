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
