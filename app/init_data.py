import csv
from .movie import Movie


def get_movies_data(path):
    movies = []

    with open(path + '/movies.csv', 'r') as movies_file:
        reader = csv.reader(movies_file, delimiter=',', )
        next(reader)  # skip header

        for row in reader:
            movies\
                .append(Movie(row[0], row[1], str(row[2]).replace('|', ' | '), path + '/images/' + str(row[0]) + '/posters/1.jpg'))

    return movies
