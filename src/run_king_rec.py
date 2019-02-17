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
item_id_page_id = 'item_page_id.csv'
dataset_path = '../data/raw/ml-latest-small/dataset/'

# define dataset keys
ratings = 'ratings'
movies = 'movies'
links = 'links'
posters = 'posters'

# define other params
epochs = 300
k = 10
n_components = 10

# read and prepare dataset
internal_user_ids, internal_item_ids = create_internal_ids(dataset_path + file_ratings)

n_items = len(internal_item_ids)
n_users = len(internal_user_ids)

raw_ratings = read_ratings(dataset_path + file_ratings, internal_user_ids, internal_item_ids)
train_interactions, test_interactions = \
    create_test_train_interactions(raw_ratings, n_users, n_items)

movie_titles_by_internal_id, movie_genres_by_internal_id = \
    read_genres(dataset_path + file_movies, internal_item_ids)
movie_genre_features = create_genres_features(movie_genres_by_internal_id, n_items)

user_features = sparse.identity(n_users)
full_item_features = sparse.hstack([sparse.identity(n_items), movie_genre_features])

# just to download posters
# movies_url_ids_by_internal_id = read_tmdb_links(dataset_path + file_links, internal_item_ids)
# save_internal_item_id_to_page_id(movies_url_ids_by_internal_id, dataset_path + item_id_page_id)
# get_tmdb_posters(dataset_path + posters + "/", movies_url_ids_by_internal_id)

# create a King Recommender model
recommender = KingRecommender(n_components)

# let's run :)
recommender.fit(user_features, full_item_features, train_interactions, epochs)
recommender.recall_at_k(user_features, full_item_features, test_interactions, k)
recommender.precision_at_k(user_features, full_item_features, test_interactions, k)
# recommender.recommend_for_user(432, movie_titles_by_internal_id, user_features, full_item_features, k)
