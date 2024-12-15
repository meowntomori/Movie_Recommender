import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy, KNNBasic
from surprise.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import requests
import os
from tqdm import tqdm
import re
import joblib
import sklearn
from packaging import version
import logging
import json
import sqlite3

logging.basicConfig(level=logging.DEBUG)

class MovieRecommender:
    def __init__(self, ratings_path='data/movie_votes.csv', movies_path='data/movie_info.csv',
                 stuff_path='data/stuff.csv', tags_path='data/tags.csv', limit=100000,
                 db_path='data/user_interactions.db'):
        logging.debug("Initializing MovieRecommender")
        self.db_path = db_path
        try:
            self.ratings = pd.read_csv(ratings_path, sep='|', nrows=limit)
        except Exception as e:
            logging.error(f'Error reading ratings csv: {e}')

        try:
            self.movies = pd.read_csv(movies_path, sep='|')
        except Exception as e:
            logging.error(f'Error reading movies csv: {e}')

        try:
            self.stuff = pd.read_csv(stuff_path, sep='|')
        except Exception as e:
            logging.error(f'Error reading stuff csv: {e}')

        try:
            self.tags = pd.read_csv(tags_path, sep='|')
            self.tags['tags'] = self.tags['tags'].fillna('').apply(
                lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')

        except Exception as e:
            logging.error(f'Error reading tags csv: {e}')

        self.movies['overview'] = self.movies['title_russian'].fillna('') + ' ' + self.movies['title_original'].fillna(
            '') + ' ' + self.movies['genre'].fillna('').apply(
            lambda x: ' '.join(eval(x)) if isinstance(x, str) else '') + ' ' + self.movies['countries'].fillna(
            '').apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')
        for col in ['actors', 'director', 'producer', 'writer', 'operator', 'composer', 'editor']:
            if col in self.movies.columns and col in self.stuff.columns:
                self.movies = pd.merge(self.movies, self.stuff[['id', col]], on='id', how='left')
                if col in self.movies.columns:
                    self.movies[col] = self.movies[col].fillna(0).astype(int).astype(str)
                    self.movies['overview'] += ' ' + self.movies[col].apply(lambda x: ' '.join(x))
            elif col in self.movies.columns:
                self.movies['overview'] += ' ' + self.movies[col].fillna('').apply(
                    lambda x: ' '.join(eval(x)) if isinstance(x, str) else x)
        self.movies = pd.merge(self.movies, self.tags, on='id', how='left')
        self.movies['overview'] = self.movies['overview'].fillna('') + ' ' + self.movies['tags'].fillna('')

        self.user_interactions = self._load_user_interactions()

        if os.path.exists('models/collaborative_model.pkl'):
            logging.debug("Loading collaborative model")
            self.model_cf = joblib.load('models/collaborative_model.pkl')
        else:
            logging.debug("Training collaborative model")
            self.train_collaborative()

        if os.path.exists('models/content_tfidf.pkl'):
            logging.debug("Loading content model")
            self.tfidf = joblib.load('models/content_tfidf.pkl')
            if self.tfidf:
                logging.debug("Creating tfidf matrix")
                self.tfidf_matrix = self.tfidf.fit_transform(self.movies['overview'])
                logging.debug(f"TF-IDF matrix loaded with shape {self.tfidf_matrix.shape}")
            else:
                logging.debug("TF-IDF is None after load")
        else:
            logging.debug("Training content model")
            self.tfidf = None
            self.tfidf_matrix = None
            self.train_content()

    def _preprocess_text(self, text):
        morph = MorphAnalyzer()
        stop_words = stopwords.words("russian")
        text = re.sub(r'[^\w\s]', '', text).lower()
        words = [morph.parse(word)[0].normal_form for word in text.split() if word not in stop_words]
        return " ".join(words)

    def _create_db_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            logging.debug("Successfully connected to db")
        except Exception as e:
            logging.error(f"Error connecting to db: {e}")
        return conn

    def _create_db_table(self):
        conn = self._create_db_connection()
        if not conn:
            return
        cursor = conn.cursor()
        try:
            cursor.execute('''
                  CREATE TABLE IF NOT EXISTS user_interactions (
                      user_id INTEGER,
                      movie_id INTEGER
                  )
              ''')
            conn.commit()
            logging.debug("Table 'user_interactions' created")
        except Exception as e:
            logging.error(f"Error creating table: {e}")
        finally:
            conn.close()

    def _load_user_interactions(self):
        self._create_db_table()
        conn = self._create_db_connection()
        if not conn:
            return {}
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT user_id, movie_id FROM user_interactions")
            rows = cursor.fetchall()
            user_interactions = {}
            for row in rows:
                user_id, movie_id = row
                if user_id not in user_interactions:
                    user_interactions[user_id] = []
                user_interactions[user_id].append(movie_id)
            return user_interactions
        except Exception as e:
            logging.error(f"Error loading user interactions from db: {e}")
            return {}
        finally:
            conn.close()

    def _save_user_interactions(self):
        conn = self._create_db_connection()
        if not conn:
            return
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM user_interactions")
            for user_id, movie_ids in self.user_interactions.items():
                for movie_id in movie_ids:
                    cursor.execute("INSERT INTO user_interactions (user_id, movie_id) VALUES (?, ?)",
                                   (user_id, movie_id))
            conn.commit()
            logging.debug("User interactions saved to database")
        except Exception as e:
            logging.error(f"Error saving user interactions to database: {e}")
        finally:
            conn.close()

    def train_content(self):
        logging.debug("Start train content")

        self.tfidf = TfidfVectorizer(stop_words=stopwords.words('russian'))
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['overview'])
        logging.debug('Trained content model')

    def train_collaborative(self, algorithm='SVD', n_factors=100, max_iter=100):
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(self.ratings[['user_id', 'movie_id', 'score']], reader)
        trainset, _ = train_test_split(data, test_size=.25)

        if algorithm == 'SVD':
            self.model_cf = SVD(n_epochs=max_iter, n_factors=n_factors)
        elif algorithm == 'KNN':
            self.model_cf = KNNBasic()
        try:
            with tqdm(desc='Training collaborative model') as progress:
                self.model_cf.fit(trainset)
        except Exception as e:
            logging.error(f'Error during collaborative training: {e}, fallback to training without tqdm')
            self.model_cf.fit(trainset)
        logging.debug('Trained collaborative model')

    def train_implicit(self, factors=20, regularization=0.01, iterations=20):
        pass

    def recommend_cf(self, user_id, num_recommendations=10, viewed_movie_id=None):
        try:
            logging.debug(f'Starting Collaborative filtering with user id: {user_id}')
            if user_id not in self.user_interactions:
                self.user_interactions[user_id] = []
            if viewed_movie_id:
                self.user_interactions[user_id].append(int(viewed_movie_id))
                self._save_user_interactions()

            user_ratings = self.ratings[self.ratings['user_id'] == user_id]
            rated_movie_ids = user_ratings['movie_id'].values
            viewed_movie_ids = self.user_interactions.get(user_id, [])

            predictions = []
            for movie_id in self.ratings['movie_id'].unique():
                if movie_id not in rated_movie_ids and movie_id not in viewed_movie_ids:
                    score = self.model_cf.predict(user_id, movie_id).est
                    predictions.append((movie_id, score))
            cf_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:num_recommendations]

            cf_movie_ids = [movie_id for movie_id, _ in cf_recommendations]
            logging.debug(f'Collaborative filtering recommendations {cf_movie_ids}')
            movie_info = self.get_movie_info(cf_movie_ids)
            return movie_info

        except Exception as e:
            logging.error(f"Collaborative filtering error: {e}")
            return []

    def recommend_implicit(self, user_id, num_recommendations=10):
        return []

    def recommend_content(self, user_query, top_n=10):
        if self.tfidf_matrix is None or self.tfidf is None:
            logging.debug("TF-IDF model not trained yet")
            return []
        logging.debug(f'Starting content based filtering with user query: {user_query}')

        query_vec = self.tfidf.transform([self._preprocess_text(user_query)])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_n]
        movie_ids = self.movies.iloc[top_indices]['id'].to_list()
        movie_info = self.get_movie_info(movie_ids)
        logging.debug(f'Content based recommendations: {movie_ids}')
        return movie_info

    def get_movie_info(self, movie_ids):
        try:
            logging.debug(f'Getting movie info for movie ids: {movie_ids}')
            movie_info = self.movies[self.movies['id'].isin(movie_ids)]
            logging.debug(f'Movie info: {movie_info}')
            return movie_info.to_dict(orient='records')
        except Exception as e:
            logging.error(f'Error finding movie info: {e}')
            return []

    def update_recommendations(self, user_id, viewed_movie_id, user_query=None):
        # Обновление данных о просмотренных фильмах
        self.user_interactions[user_id].append(viewed_movie_id)
        self._save_user_interactions()

        # Переобучение моделей
        self.train_collaborative()
        self.train_content()

        # Обновление рекомендаций
        cf_recommendations = self.recommend_cf(user_id)

        # Для контент-базированных рекомендаций используем строку запроса
        if user_query:
            content_recommendations = self.recommend_content(user_query)
        else:
            # Если строка запроса не предоставлена, используем последний просмотренный фильм
            last_viewed_movie = self.movies[self.movies['id'] == viewed_movie_id]['overview'].values[0]
            content_recommendations = self.recommend_content(last_viewed_movie)

        logging.debug(f"CF Recommendations: {cf_recommendations}")
        logging.debug(f"Content Recommendations: {content_recommendations}")

        return cf_recommendations, content_recommendations

    def get_top_movies(self, top_n=5):
        # Возвращаем топовые фильмы
        top_movies = self.movies.head(top_n)
        return self.format_movies(top_movies)

    def get_additional_movies(self, top_n=50):
        # Возвращаем дополнительные фильмы
        additional_movies = self.movies.head(top_n)
        return self.format_movies(additional_movies)

    def format_movies(self, movies):
        unique_movies = []
        seen_titles = set()

        for _, movie in movies.iterrows():
            title = movie['title_russian']
            genres = movie['genre']
            if title not in seen_titles:
                seen_titles.add(title)
                formatted_genres = ', '.join(eval(genres)) if isinstance(genres, str) else ', '.join(genres)
                unique_movies.append({
                    'id': movie['id'],
                    'title_russian': title,
                    'genre': formatted_genres
                })

        return unique_movies

    def update_user_interactions(self, user_id, viewed_movie_id):
        logging.debug(f"Updating user interactions for user_id: {user_id}, viewed_movie_id: {viewed_movie_id}")
        self.user_interactions[user_id].append(viewed_movie_id)
        self._save_user_interactions()

    def recommend_cf(self, user_id, num_recommendations=10, viewed_movie_id=None):
        logging.debug(f"Starting Collaborative filtering with user id: {user_id}")
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []
        if viewed_movie_id:
            self.user_interactions[user_id].append(int(viewed_movie_id))
            self._save_user_interactions()

        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        rated_movie_ids = user_ratings['movie_id'].values
        viewed_movie_ids = self.user_interactions.get(user_id, [])  # получаем просмотренные фильмы

        predictions = []
        for movie_id in self.ratings['movie_id'].unique():
            if movie_id not in rated_movie_ids and movie_id not in viewed_movie_ids:  # убираем просмотренные
                score = self.model_cf.predict(user_id, movie_id).est
                predictions.append((movie_id, score))
        # сортируем и берем первые n
        cf_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:num_recommendations]

        cf_movie_ids = [movie_id for movie_id, _ in cf_recommendations]
        logging.debug(f"Collaborative filtering recommendations {cf_movie_ids}")
        movie_info = self.get_movie_info(cf_movie_ids)
        return movie_info

    def get_movie_info(self, movie_ids):
        try:
            logging.debug(f'Getting movie info for movie ids: {movie_ids}')
            movie_info = self.movies[self.movies['id'].isin(movie_ids)]
            logging.debug(f'Movie info: {movie_info}')

            formatted_movies = []
            seen_titles = set()

            for _, movie in movie_info.iterrows():
                title = movie['title_russian']
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                genres = movie['genre']
                formatted_genres = ' '.join(genre.capitalize() for genre in eval(genres)) if isinstance(genres,
                                                                                                        str) else ''
                formatted_movies.append({
                    'id': movie['id'],
                    'title_russian': title,
                    'genre': formatted_genres,
                    'rating_imdb_value': movie['rating_imdb_value']
                })

            return formatted_movies
        except Exception as e:
            logging.error(f'Error finding movie info: {e}')
            return []


