import os
import pandas as pd
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import nltk

nltk.download('stopwords', quiet=True)

# Путь к директории с данными
data_dir = 'data/reviews'

# Получение списка всех файлов
files = []
for root, dirs, filenames in os.walk(data_dir):
    for filename in filenames:
        files.append(os.path.join(root, filename))

# Инициализация списка для хранения данных
data_list = []

# Чтение всех файлов и объединение данных в DataFrame
for file_path in tqdm(files, desc='Reading files'):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        movie_id = os.path.basename(file_path).split('-')[0]
        review_number = os.path.basename(file_path).split('-')[1].split('.')[0]
        sentiment = os.path.basename(os.path.dirname(file_path))
        data_list.append({'file_path': file_path, 'text': text, 'movie_id': movie_id,
                          'review_number': review_number, 'sentiment': sentiment})

data = pd.DataFrame(data_list)

# Инициализация MorphAnalyzer один раз
morph = MorphAnalyzer()

def preprocess_text(text):
    stop_words = stopwords.words("russian")
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = [morph.parse(word)[0].normal_form for word in text.split() if word not in stop_words]
    return " ".join(words)

data['processed_text'] = data['text'].apply(preprocess_text)

# Сохранение предобработанных данных в CSV файл
data.to_csv('data/preprocessed_reviews.csv', index=False)
