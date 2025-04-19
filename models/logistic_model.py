from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib

class SentimentAnalyzer:
    def __init__(self, model_type='logistic'):
        """
        Инициализация модели.
        :param model_type: 'logistic' для логистической регрессии или 'naive_bayes' для наивного Байеса
        """
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        else:
            raise ValueError("model_type must be 'logistic' or 'naive_bayes'")
        self.vectorizer = None
    
    def fit(self, X, y, vectorizer):
        """
        Обучение модели и векторизатора.
        :param X: список текстов
        :param y: метки классов
        :param vectorizer: объект TextVectorizer
        :return: self для цепочки вызовов
        """
        self.vectorizer = vectorizer
        X_transformed = self.vectorizer.fit_transform(X)
        self.model.fit(X_transformed, y)
        return self
    
    def predict(self, X):
        """
        Предсказание классов для новых текстов.
        :param X: список текстов
        :return: предсказанные классы
        """
        X_transformed = self.vectorizer.transform(X)
        return self.model.predict(X_transformed)
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей для новых текстов.
        :param X: список текстов
        :return: вероятности для каждого класса
        """
        X_transformed = self.vectorizer.transform(X)
        return self.model.predict_proba(X_transformed)
    
    def save(self, model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
        """
        Сохранение модели и векторизатора.
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
    
    def load(self, model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
        """
        Загрузка модели и векторизатора.
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        return self