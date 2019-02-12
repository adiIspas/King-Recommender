import csv
import logging
from collections import defaultdict

from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer


def read_ratings(file_path):
    logging.info("Loading ratings ...")

    with open(file_path, 'r') as ratings_file:
        ratings_file_reader = csv.reader(ratings_file)
        raw_ratings = list(ratings_file_reader)
        raw_ratings.pop(0)  # remove header

    movielens_to_internal_user_ids = defaultdict(lambda: len(movielens_to_internal_user_ids))
    movielens_to_internal_item_ids = defaultdict(lambda: len(movielens_to_internal_item_ids))

    for row in raw_ratings:
        row[0] = movielens_to_internal_user_ids[int(row[0])]
        row[1] = movielens_to_internal_item_ids[int(row[1])]
        row[2] = float(row[2])

    n_users = len(movielens_to_internal_user_ids)
    n_items = len(movielens_to_internal_item_ids)

    return raw_ratings, n_users, n_items, movielens_to_internal_item_ids


def read_genres(file_path, movielens_to_internal_item_ids):
    logging.info("Loading generes ...")

    n_items = len(movielens_to_internal_item_ids)

    with open(file_path, 'r') as genres_file:
        genres_file_reader = csv.reader(genres_file)
        raw_genres = list(genres_file_reader)
        raw_genres.pop(0)  # remove header

    movie_genres_by_internal_id = {}
    movie_titles_by_internal_id = {}

    for row in raw_genres:
        row[0] = movielens_to_internal_item_ids[int(row[0])]
        row[2] = row[2].split('|')

        movie_titles_by_internal_id[row[0]] = row[1]
        movie_genres_by_internal_id[row[0]] = row[2]

    # Build a list of genres where the index is the internal movie ID and
    # the value is a list of [Genre, Genre, ...]
    movie_genres = [movie_genres_by_internal_id[internal_id]
                    for internal_id in range(n_items)]

    # Transform the genres into binarized labels using scikit's MultiLabelBinarizer
    movie_genre_features = MultiLabelBinarizer().fit_transform(movie_genres)
    n_genres = movie_genre_features.shape[1]

    # Coerce the movie genre features to a sparse matrix, which TensorRec expects
    movie_genre_features = sparse.coo_matrix(movie_genre_features)

    return movie_genre_features, n_genres, movie_titles_by_internal_id
