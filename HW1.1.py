from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import torch
import math

# Загрузка предобученной модели и токенайзера
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Устанавливаем pad_token_id для корректной работы
tokenizer.pad_token = tokenizer.eos_token

# Текстовый ввод для генерации
prompt = "Science and technology of the future"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Генерация текста
output = model.generate(
    input_ids,
    max_length=100,  # Максимальная длина текста
    num_return_sequences=1,  # Количество генерируемых текстов
    temperature=0.7,  # Контроль за случайностью
    top_p=0.9,  # Накопительная вероятность для фильтрации токенов
    do_sample=True,  # Включение семплирования
    pad_token_id=tokenizer.pad_token_id  # Указываем идентификатор заполнителя
)

# Декодирование текста
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Сгенерированный текст:")
print(generated_text)

# ===== Метрика 1: Перплексия =====
def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    perplexity = math.exp(loss.item())
    return perplexity

perplexity = calculate_perplexity(model, tokenizer, generated_text)
print(f"Перплексия сгенерированного текста: {perplexity:.2f}")

# ===== Метрика 2: BLEU =====
# Пример референсного текста
reference_text = "The science and technology of the future involves artificial intelligence and robotics."

# Токенизация референсного текста и сгенерированного текста
reference_tokens = reference_text.split()
generated_tokens = generated_text.split()

# BLEU оценка
bleu_score = sentence_bleu([reference_tokens], generated_tokens)
print(f"BLEU-оценка: {bleu_score:.4f}")

# ===== Метрика 3: ROUGE =====
def calculate_rouge(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

rouge_scores = calculate_rouge(generated_text, reference_text)
print("ROUGE оценки:")
for key, value in rouge_scores.items():
    print(f"  {key}: Precision: {value.precision:.4f}, Recall: {value.recall:.4f}, F1: {value.fmeasure:.4f}")
