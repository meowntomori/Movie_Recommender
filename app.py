from flask import Flask, render_template, request, jsonify, session
from recommender import MovieRecommender
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = '937221283944d3a47553a81075de396e28a24f48aab301b24d64e2b7e36b31c7'  # Замените на ваш секретный ключ
recommender = MovieRecommender()

@app.route("/", methods=['GET'])
def welcome():
    return render_template("welcome.html")

@app.route("/search", methods=['GET', 'POST'])
def index():
    recommendations = []
    method = 'cf'
    error = None
    user_id = session.get('user_id')
    search_query = None

    if request.method == 'POST':
        user_id = request.form.get('user_id')
        search_query = request.form.get('search_query')
        method = request.form.get('method')
        viewed_movie_id = request.form.get('viewed_movie_id')

        if user_id:
            session['user_id'] = user_id
            try:
                if method == 'cf':
                    recommendations = recommender.recommend_cf(int(user_id), viewed_movie_id=viewed_movie_id if viewed_movie_id else None)
                elif method == 'content':
                    recommendations = recommender.recommend_content(search_query)
            except ValueError:
                error = "Пожалуйста, введите корректный ID пользователя."
            except Exception as e:
                error = f"Произошла ошибка: {e}"
    elif request.method == 'GET' and request.args.get('user_id'):
        user_id = request.args.get('user_id')
        search_query = request.args.get('search_query')
        method = request.args.get('method')
        try:
            if method == 'cf':
                recommendations = recommender.recommend_cf(int(user_id))
            elif method == 'content':
                recommendations = recommender.recommend_content(search_query)
        except ValueError:
            error = "Пожалуйста, введите корректный ID пользователя."
        except Exception as e:
            error = f"Произошла ошибка: {e}"
    return render_template("index.html", recommendations=recommendations, method=method, error=error, user_id=user_id, search_query=search_query)

@app.route("/update_recommendations", methods=['POST'])
def update_recommendations():
    user_id = session.get('user_id')
    viewed_movie_id = request.form.get('viewed_movie_id')
    search_query = request.form.get('user_query')
    cf_recommendations, content_recommendations = recommender.update_recommendations(int(user_id), int(viewed_movie_id), search_query)
    return jsonify({
        'cf_recommendations': cf_recommendations,
        'content_recommendations': content_recommendations
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
