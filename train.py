from models.logistic_model import SentimentAnalyzer
from preprocessing.text_cleaner import TextCleaner
from preprocessing.tokenizer_vectorizer import TextVectorizer
from metrics.evaluation import ModelEvaluator
from utils.io import DataIO
from sklearn.model_selection import train_test_split, cross_val_score

def main():
    # Загрузка данных из .csv файла
    data = DataIO.load_data(r'C:\Users\USER\Desktop\SRS3\youtube_comments.csv')
    
    # Проверка столбцов
    print("Столбцы в данных:", data.columns)
    print(data[['Comment', 'Sentiment']].head())
    
    # Обработка пропущенных значений
    data = data.dropna(subset=['Comment', 'Sentiment'])  # Удаляем строки с NaN
    
    # Извлечение данных
    comments = data['Comment']
    sentiments = data['Sentiment']
    
    # Предобработка
    cleaner = TextCleaner()
    cleaned_comments = [cleaner.clean_text(comment) for comment in comments]
    
    # Векторизация (выбор между 'bow' и 'tfidf')
    vectorizer = TextVectorizer(vectorizer_type='tfidf')  # Можно сменить на 'bow'
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(cleaned_comments, sentiments, test_size=0.2, random_state=42)
    
    # Обучение модели (выбор между 'logistic' и 'naive_bayes')
    analyzer = SentimentAnalyzer(model_type='logistic')  # Можно сменить на 'naive_bayes'
    
    # Кросс-валидация
    X_train_transformed = vectorizer.fit_transform(X_train)
    scores = cross_val_score(analyzer.model, X_train_transformed, y_train, cv=5, scoring='f1_macro')
    print("Кросс-валидация F1 (macro):", scores.mean(), "+/-", scores.std())
    
    # Финальное обучение на всех тренировочных данных
    analyzer.fit(X_train, y_train, vectorizer)
    
    # Оценка на тестовых данных
    y_pred = analyzer.predict(X_test)
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred)
    evaluator.plot_confusion_matrix(y_test, y_pred)
    evaluator.plot_feature_importance(analyzer, vectorizer)
    
    # Сохранение результатов
    DataIO.save_results(metrics, r'C:\Users\USER\Desktop\SRS3\evaluation_results.txt')
    
    # Сохранение модели и векторизатора
    analyzer.save('sentiment_model.pkl', 'vectorizer.pkl')
    print("Модель и векторизатор сохранены в 'sentiment_model.pkl' и 'vectorizer.pkl'")

if __name__ == "__main__":
    main()