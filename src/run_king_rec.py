import logging
import os

from scipy import sparse

from src.modeling.recommender import KingRecommender
from src.preparation.csv_reader import read_ratings, read_genres, create_internal_ids
from src.processing.data import create_test_train_interactions, create_genres_features

# this is set to use CPU instead of GPU -> CUDA driver problem
os.environ['CUDA_VISIBLE_DEVICES'] = ''
logging.getLogger().setLevel(logging.INFO)

# define file paths
file_ratings = 'ratings.csv'
file_movies = 'movies.csv'
file_links = 'links.csv'
dataset_path = '../data/raw/ml-latest-small/dataset/'

# define dataset keys
ratings = 'ratings'
movies = 'movies'
links = 'links'

# read and prepare dataset
internal_user_ids, internal_item_ids = create_internal_ids(dataset_path + file_ratings)

n_items = len(internal_item_ids)
n_users = len(internal_user_ids)

raw_ratings = read_ratings(dataset_path + file_ratings, internal_user_ids, internal_item_ids)
sparse_train_ratings_4plus, sparse_test_ratings_4plus = \
    create_test_train_interactions(raw_ratings, n_users, n_items)

movie_titles_by_internal_id, movie_genres_by_internal_id = \
    read_genres(dataset_path + file_movies, internal_item_ids)
movie_genre_features = create_genres_features(movie_genres_by_internal_id, n_items)

user_features = sparse.identity(n_users)
full_item_features = sparse.hstack([sparse.identity(n_items), movie_genre_features])

# create a King Recommender model
recommender = KingRecommender(user_features, full_item_features, sparse_train_ratings_4plus, sparse_test_ratings_4plus)

# let's run :)
recommender.fit(10)
recommender.recall_at_k()
recommender.recommend_for_user(432, movie_titles_by_internal_id, 5)
