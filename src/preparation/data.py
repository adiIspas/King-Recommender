import logging

from scipy import sparse
from sklearn.model_selection import train_test_split


def split_data(data, train_size=0.8):
    logging.info("Split data into %s%% train and %s%% test", train_size * 100, 100 - train_size * 100)
    return train_test_split(data, train_size=train_size, test_size=1 - train_size)


def interactions_list_to_sparse_matrix(interactions, n_users, n_items):
    logging.info("Create sparse matrix for %s users and %s items", n_users, n_items)
    users_column, items_column, ratings_column, _ = zip(*interactions)
    return sparse.coo_matrix((ratings_column, (users_column, items_column)), shape=(n_users, n_items))
