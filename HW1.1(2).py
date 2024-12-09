from transformers import pipeline

# Инициализация пайплайна
sentiment_model = pipeline("sentiment-analysis")

# Пример текста для анализа
text = [
    "I absolutely love this movie, it was fantastic!",
    "The food at the restaurant was terrible, I will never go back.",
    "What a beautiful day to go for a walk in the park!",
    "The customer service was awful, they didn’t help me at all.",
    "This product exceeded my expectations, highly recommend it.",
    "I can’t believe how rude that person was to me.",
    "The concert last night was an unforgettable experience.",
    "I am so disappointed with the quality of this item.",
    "The view from the top of the mountain was breathtaking.",
    "Traffic today was horrible, I was late for work.",
    "I feel so grateful for all the support I’ve received.",
    "This software is extremely buggy and difficult to use.",
    "I had such a wonderful time catching up with my old friends.",
    "The hotel room was dirty and smelled terrible.",
    "I’m so excited about the new project we’re starting.",
    "I hate when people are late, it’s so disrespectful.",
    "The book I just read was so inspiring and motivational.",
    "My flight got delayed for hours, what a nightmare!",
    "I really appreciate how attentive and kind the staff were.",
    "The weather has been so unpredictable lately, it’s frustrating.",
    "Weapons in WARFRAME obtain Affinity (experience points) when used in combat.",
    "A weapon's Mod capacity is equal to the higher of either the weapon's rank or the player's Mastery Rank.",
    "An Orokin Catalyst can be installed to double the weapon's capacity, for a new maximum of 60.",
    "A new player will have the choice of 6 different weapons in the Awakening tutorial, 2 from each weapon slot.",
    "Most weapons can be obtained by crafting them in the Foundry with a blueprint and resources/components.",
    "The Old War found humanity facing a technologically superior force, and their weapons were turned against them.",
    "Melee and ballistic weapons, inspired by primitive counterparts, became part of the Tenno arsenal to circumvent the Sentient interference of more technologically-involved weaponry.",
    "A player's loadout can have up to 3 weapons; primary, secondary, and melee.",
    "Primary weapons are mainly separated into rifles, shotguns, sniper rifles, and bows, as these are the four categories that have unique mods to them, as well as Sortie missions limited to each type."
]


# Анализ тональности
results = sentiment_model(text)

# Вывод результата
for idx, result in enumerate(results):
    print(f"Text {idx + 1}: {text[idx]}")
    print(f"Label: {result['label']}, Score: {result['score']:.2f}")
