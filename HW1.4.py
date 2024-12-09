import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

# Загрузка модели и процессора
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Функция для детекции объектов на одном кадре
def detect_objects_on_frame(frame):
    # Конвертация кадра в формат PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Подготовка входных данных
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Постпроцессинг
    target_sizes = torch.tensor([image.size[::-1]])  # Размер изображения (H, W)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)
    
    # Отрисовка объектов на кадре
    for i, result in enumerate(results):
        for box, score, label in zip(result['boxes'], result['scores'], result['labels']):
            if score > 0.9:  # Порог вероятности
                box = box.int().tolist()
                # Отрисовка рамки и метки
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                label_text = f"{model.config.id2label[label.item()]}: {score:.2f}"
                cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Загрузка видео
video_path = "video1.mp4"  # Замените на путь к вашему видео
cap = cv2.VideoCapture(video_path)

# Проверка, открылось ли видео
if not cap.isOpened():
    print("Не удалось открыть видео.")
    exit()

# Подготовка для записи результата
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Обработка видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Детекция объектов на текущем кадре
    frame_with_objects = detect_objects_on_frame(frame)
    
    # Запись обработанного кадра
    out.write(frame_with_objects)
    # (Опционально) Показываем кадры в реальном времени
    cv2.imshow("Detection", frame_with_objects)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Завершение работы
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Обработанное видео сохранено в {output_path}.")
