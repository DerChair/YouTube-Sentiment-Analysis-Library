import pandas as pd

# Загрузка данных из .xlsx файла
data = pd.read_excel(r"D:\YoutubeCommentsDataSet.xlsx", engine='openpyxl')

# Разделение столбца Comment_Sentiment на два: Comment и Sentiment
data[['Comment', 'Sentiment']] = data['Comment,Sentiment'].str.rsplit(',', n=1, expand=True)

# Удаление лишних пробелов
data['Comment'] = data['Comment'].str.strip()
data['Sentiment'] = data['Sentiment'].str.strip()

# Удаление исходного столбца Comment_Sentiment
data = data.drop(columns=['Comment,Sentiment'])

# Сохранение результата в .csv файл
data.to_csv(r'C:\Users\USER\Desktop\SRS3\youtube_comments.csv', index=False)

print("Файл успешно конвертирован в youtube_comments_processed.csv")
print("Первые строки данных:")
print(data.head())