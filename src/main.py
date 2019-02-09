import logging

from src.preparation.csv_reader import read_ratings
from src.preparation.data import split_data, interactions_list_to_sparse_matrix

logging.getLogger().setLevel(logging.INFO)

file_path = "../data/raw/ml-latest-small/dataset/ratings.csv"
raw_ratings, n_users, n_items = read_ratings(file_path)

train_ratings, test_ratings = split_data(raw_ratings, train_size=0.8)

sparse_train_ratings = interactions_list_to_sparse_matrix(train_ratings, n_users, n_items)
sparse_test_ratings = interactions_list_to_sparse_matrix(test_ratings, n_users, n_items)
