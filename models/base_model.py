# youtube_sentiment_lib/models/base_model.py
from abc import ABC, abstractmethod

class BaseSentimentModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        pass

