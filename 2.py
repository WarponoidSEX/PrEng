import speech_recognition as sr

# Инициализация распознавателя речи
recognizer = sr.Recognizer()

# Инициализация микрофона
microphone = sr.Microphone()

# Функция для захвата и распознавания речи с микрофона
def recognize_speech_from_microphone():
    with microphone as source:
        print("Настройка микрофона, пожалуйста, подождите...")
        recognizer.adjust_for_ambient_noise(source)  # Устанавливаем порог шума
        print("Слушаю, говорите...")
        
        # Запись аудио
        audio_data = recognizer.listen(source)
        print("Распознаю...")
        
        try:
            # Используем Google Web Speech API для распознавания речи
            text = recognizer.recognize_google(audio_data, language="ru-RU")
            print("Распознанный текст:", text)
        except sr.UnknownValueError:
            print("Не удалось распознать аудио")
        except sr.RequestError as e:
            print(f"Ошибка запроса; {e}")

# Запуск функции для распознавания речи
recognize_speech_from_microphone()
