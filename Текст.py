from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка предобученной модели и токенайзера
model_name = "gpt2"  
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Устанавливаем pad_token_id для корректной работы
tokenizer.pad_token = tokenizer.eos_token

# Текстовый ввод для генерации
prompt = "Science and technology of the future"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Добавляем attention_mask
attention_mask = input_ids != tokenizer.pad_token_id

# Генерация текста
output = model.generate(
    input_ids,
    attention_mask=attention_mask,  # Указываем маску внимания
    max_length=500,  # Максимальная длина текста
    num_return_sequences=1,  # Количество генерируемых текстов
    temperature=0.7,  # Контроль за случайностью
    top_p=0.9,  # Накопительная вероятность для фильтрации токенов
    do_sample=True,  # Включение семплирования
    pad_token_id=tokenizer.pad_token_id  # Указываем идентификатор заполнителя
)

# Декодирование и вывод результата
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Сгенерированный текст:")
print(generated_text)
