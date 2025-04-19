from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class TextVectorizer:
    def __init__(self, vectorizer_type='bow'):
        """
        Инициализация векторизатора.
        :param vectorizer_type: 'bow' для Bag of Words или 'tfidf' для TF-IDF
        """
        if vectorizer_type == 'bow':
            self.vectorizer = CountVectorizer()
        elif vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        else:
            raise ValueError("vectorizer_type must be 'bow' or 'tfidf'")
    
    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()