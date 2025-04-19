from models.logistic_model import SentimentAnalyzer

def predict_comment(comment):
    # Загрузка модели и векторизатора
    analyzer = SentimentAnalyzer()
    analyzer.load('sentiment_model.pkl', 'vectorizer.pkl')
    
    # Предсказание для нового комментария
    prediction = analyzer.predict([comment])[0]
    probabilities = analyzer.predict_proba([comment])[0]
    
    # Формирование результата
    labels = ['negative', 'neutral', 'positive']
    result = {
        'comment': comment,
        'sentiment': prediction,
        'probabilities': dict(zip(labels, probabilities))
    }
    return result

if __name__ == "__main__":
    comment = "This phone is amazing!"
    result = predict_comment(comment)
    print(result)