from flask import Flask, render_template, request
from app import init_data

app = Flask(__name__)
dataset = '/home/adi/Coding/Unibuc/king-rec-dataset/ml-latest-small'


@app.route("/")
def hello(last_movie_id=0):
    movies = init_data.get_movies_data(dataset)[last_movie_id:last_movie_id + 5]
    return render_template('index.html', username='Adi', movies=movies)


@app.route("/like")
def select_a_movie():
    print('Selected movie with id:', request.args.get('movie_id'))

    last_movie_id = int(request.args.get('movie_id'))
    return hello(last_movie_id)


app.jinja_env.globals['select_a_movie'] = select_a_movie
