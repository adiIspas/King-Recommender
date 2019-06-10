from flask import Flask, render_template
from app import init_data

app = Flask(__name__)
dataset = '/home/adi/Coding/Unibuc/king-rec-dataset/ml-latest-small'


@app.route("/")
def hello():
    movies = init_data.get_movies_data(dataset)[0:10]
    return render_template('index.html', username='Adi', movies=movies)
