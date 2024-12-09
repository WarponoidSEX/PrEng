from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Загрузка модели и токенизатора
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Функция для анализа тональности
def analyze_sentiment(texts):
    """
    Анализ тональности текстов с использованием модели Hugging Face.
    :param texts: строка или список строк
    :return: список с результатами анализа тональности
    """
    if isinstance(texts, str):
        texts = [texts]

    results = []
    for text in texts:
        # Токенизация текста
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Прогноз с помощью модели
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Определение метки и вероятности
        label = "POSITIVE" if torch.argmax(probabilities) == 1 else "NEGATIVE"
        score = probabilities[0][torch.argmax(probabilities)].item()

        # Сохранение результата
        results.append({
            "text": text,
            "label": label,
            "score": round(score, 4)
        })

    return results

# Пример использования
if __name__ == "__main__":
    sample_texts = [
        "I absolutely love this product! It's amazing!",
        "Milady, shall we go for a walk?",
        "I will do anything to kill you"
    ]

    sentiments = analyze_sentiment(sample_texts)
    for result in sentiments:
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['label']}")
        print(f"Score: {result['score']}")
        print("-" * 40)
