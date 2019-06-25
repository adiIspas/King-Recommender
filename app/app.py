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
    max_movies = 5
    user_id = init_model.user_id
    movies = init_data.get_movies_data(dataset)

    # init_model.update_model(user_id=user_id, movie_id_index=movies_ids.index(str(movie_id)))
    init_model.update_model(user_id=user_id, movie_id=movie_id)
    scores = init_model.model.predict(user_ids=[user_id], item_ids=np.arange(len(movies_ids))[0:9724])
    movies_recommended = np.array(movies_ids)[[np.argsort(-scores)]][0:max_movies]
    movies = [movie for movie in movies if movie.id in movies_recommended]

    return render_template('index.html', user_id=user_id, movies=movies)


@app.route("/like")
def select_a_movie():
    print('Selected movie with id:', request.args.get('movie_id'))

    movie_id = int(request.args.get('movie_id'))
    return load(movie_id)


@app.route("/change")
def change_user():
    # init_model.user_id = random.randrange(1, 609)
    init_model.user_id = init_model.user_id + 1

    return load()


app.jinja_env.globals['select_a_movie'] = select_a_movie

init_model = get_init_model()
movies_ids = init_data.get_movies_ids(dataset)
