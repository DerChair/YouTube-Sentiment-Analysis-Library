import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        # Проверка, является ли text строкой
        if not isinstance(text, str):
            return ""  # Если text не строка (например, NaN или float), возвращаем пустую строку
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление специальных символов и цифр
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Удаление стоп-слов
        text = ' '.join(word for word in text.split() if word not in self.stop_words)
        return text