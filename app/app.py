import numpy as np

from flask import Flask, render_template, request
from app import init_data
from app.init_model import InitModel

app = Flask(__name__)
dataset = '/home/adi/Coding/Unibuc/king-rec-dataset/ml-latest-small'


def get_init_model():
    init = InitModel()
    init.load_model()

    return init


@app.route("/")
def load(movie_id=1):
    user_id = 1
    movies = init_data.get_movies_data(dataset)

    init_model.update_model(user_id=user_id, movie_id_index=movies_ids.index(str(movie_id)))
    scores = init_model.model.predict(user_ids=[user_id], item_ids=np.arange(len(movies_ids)))
    movies_recommended = np.array(movies_ids)[[np.argsort(-scores)]][0:5]
    movies = [movie for movie in movies if movie.id in movies_recommended]

    return render_template('index.html', username='Adi', movies=movies)


@app.route("/like")
def select_a_movie():
    print('Selected movie with id:', request.args.get('movie_id'))

    movie_id = int(request.args.get('movie_id'))
    return load(movie_id)


app.jinja_env.globals['select_a_movie'] = select_a_movie

init_model = get_init_model()
movies_ids = init_data.get_movies_ids(dataset)
