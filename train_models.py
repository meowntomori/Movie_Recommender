import os
import pandas as pd
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import nltk
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, accuracy, KNNBasic
from surprise.model_selection import train_test_split as surprise_train_test_split
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)

nltk.download('stopwords', quiet=True)

def load_data(file_path, sep='|'):
    try:
        data = pd.read_csv(file_path, sep=sep)
        print(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def preprocess_text(text):
    morph = MorphAnalyzer()
    stop_words = stopwords.words("russian")
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = [morph.parse(word)[0].normal_form for word in text.split() if word not in stop_words]
    return " ".join(words)

def handle_missing_values(df):
    df.dropna(inplace=True)
    print("Handled missing values")
    return df

def convert_to_lowercase(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].str.lower()
    print("Converted columns to lowercase")
    return df

def convert_to_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print("Converted columns to numeric")
    return df

# Загрузка данных
ratings_path = 'data/movie_votes.csv'
movies_path = 'data/movie_info.csv'
stuff_path = 'data/stuff.csv'
tags_path = 'data/tags.csv'

ratings = load_data(ratings_path)
movies = load_data(movies_path)
stuff = load_data(stuff_path)
tags = load_data(tags_path)

# Обработка данных
ratings = handle_missing_values(ratings)
movies = handle_missing_values(movies)
stuff = handle_missing_values(stuff)
tags = handle_missing_values(tags)

movies = convert_to_lowercase(movies, ['title_russian', 'title_original'])
movies = convert_to_numeric(movies, ['production_year', 'duration'])

# Предобработка текста
movies['overview'] = movies['title_russian'].fillna('') + ' ' + movies['title_original'].fillna(
    '') + ' ' + movies['genre'].fillna('').apply(
    lambda x: ' '.join(eval(x)) if isinstance(x, str) else '') + ' ' + movies['countries'].fillna(
    '').apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')
for col in ['actors', 'director', 'producer', 'writer', 'operator', 'composer', 'editor']:
    if col in movies.columns and col in stuff.columns:
        movies = pd.merge(movies, stuff[['id', col]], on='id', how='left')
        if col in movies.columns:
            movies[col] = movies[col].fillna(0).astype(int).astype(str)
            movies['overview'] += ' ' + movies[col].apply(lambda x: ' '.join(x))
    elif col in movies.columns:
        movies['overview'] += ' ' + movies[col].fillna('').apply(
            lambda x: ' '.join(eval(x)) if isinstance(x, str) else x)
movies = pd.merge(movies, tags, on='id', how='left')
movies['overview'] = movies['overview'].fillna('') + ' ' + movies['tags'].fillna('')

movies['processed_text'] = movies['overview'].apply(preprocess_text)

# Разделение данных на обучающую и тестовую выборки
train_ratings, test_ratings = train_test_split(ratings, test_size=0.25)
train_movies, test_movies = train_test_split(movies, test_size=0.25)

class MovieRecommender:
    def __init__(self, train_ratings, train_movies):
        self.train_ratings = train_ratings
        self.train_movies = train_movies
        self.train_collaborative()
        self.train_content()

    def train_collaborative(self, algorithm='SVD', n_factors=100, max_iter=100):
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(self.train_ratings[['user_id', 'movie_id', 'score']], reader)
        trainset, testset = surprise_train_test_split(data, test_size=.25)

        if algorithm == 'SVD':
            self.model_cf = SVD(n_epochs=max_iter, n_factors=n_factors)
        elif algorithm == 'KNN':
            self.model_cf = KNNBasic()
        try:
            self.model_cf.fit(trainset)
        except Exception as e:
            print(f'Error during collaborative training: {e}')

        # Оценка модели
        predictions = self.model_cf.test(testset)
        rmse = accuracy.rmse(predictions)
        print(f'Collaborative model RMSE: {rmse}')
        print('Trained collaborative model')

    def train_content(self):
        self.tfidf = TfidfVectorizer(stop_words=stopwords.words('russian'))
        self.tfidf_matrix = self.tfidf.fit_transform(self.train_movies['processed_text'])
        print('Trained content model')

    def save_models(self):
        joblib.dump(self.model_cf, 'models/collaborative_model.pkl')
        joblib.dump(self.tfidf, 'models/content_tfidf.pkl')
        print('Models saved successfully')

# Обучение моделей
recommender = MovieRecommender(train_ratings, train_movies)
recommender.save_models()
