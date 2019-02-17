import logging

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


def split_data(data, train_size=0.8):
    logging.info("Split data into %s%% train and %s%% test", train_size * 100, 100 - train_size * 100)
    return train_test_split(data, train_size=train_size, test_size=1 - train_size)


def interactions_list_to_sparse_matrix(interactions, n_users, n_items):
    logging.info("Create sparse matrix for %s users and %s items", n_users, n_items)

    users_column, items_column, ratings_column, _ = zip(*interactions)
    return sparse.coo_matrix((ratings_column, (users_column, items_column)), shape=(n_users, n_items))


def create_test_train_interactions(data, n_users, n_items, train_size=0.8, min_rating=4.5):
    train_data, test_data = split_data(data, train_size=train_size)

    sparse_train_ratings = interactions_list_to_sparse_matrix(train_data, n_users, n_items)
    sparse_test_ratings = interactions_list_to_sparse_matrix(test_data, n_users, n_items)

    sparse_train_ratings_plus = sparse_train_ratings.multiply(sparse_train_ratings >= min_rating)
    sparse_test_ratings_plus = sparse_test_ratings.multiply(sparse_test_ratings >= min_rating)

    return sparse_train_ratings_plus, sparse_test_ratings_plus


def create_genres_features(movie_genres_by_internal_id, n_items):
    movie_genres = [movie_genres_by_internal_id[internal_id] for internal_id in range(n_items)]

    movie_genre_features = MultiLabelBinarizer().fit_transform(movie_genres)
    movie_genre_features = sparse.coo_matrix(movie_genre_features)

    return movie_genre_features
