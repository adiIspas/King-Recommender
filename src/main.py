import logging
import os

import numpy
import tensorrec
from scipy import sparse

from src.preparation.csv_reader import read_ratings, read_genres
from src.preparation.data import split_data, interactions_list_to_sparse_matrix

# this is set to use CPU instead of GPU -> CUDA driver problem
os.environ['CUDA_VISIBLE_DEVICES'] = ''

logging.getLogger().setLevel(logging.INFO)

file_path_ratings = "../data/raw/ml-latest-small/dataset/ratings.csv"
file_path_metadata = "../data/raw/ml-latest-small/dataset/movies.csv"
raw_ratings, n_users, n_items, movielens_to_internal_item_ids = read_ratings(file_path_ratings)

train_ratings, test_ratings = split_data(raw_ratings, train_size=0.8)

sparse_train_ratings = interactions_list_to_sparse_matrix(train_ratings, n_users, n_items)
sparse_test_ratings = interactions_list_to_sparse_matrix(test_ratings, n_users, n_items)

sparse_train_ratings_4plus = sparse_train_ratings.multiply(sparse_train_ratings >= 4.0)
sparse_test_ratings_4plus = sparse_test_ratings.multiply(sparse_test_ratings >= 4.0)

# Construct indicator features for users and items
user_indicator_features = sparse.identity(n_users)
item_indicator_features = sparse.identity(n_items)

movie_genre_features, n_genres, movie_titles_by_internal_id = read_genres(file_path_metadata,
                                                                          movielens_to_internal_item_ids=movielens_to_internal_item_ids)

# Try concatenating the genres on to the indicator features for a hybrid recommender system
full_item_features = sparse.hstack([item_indicator_features, movie_genre_features])

# Build a matrix factorization collaborative filter model
hybrid_model = tensorrec.TensorRec(n_components=5,
                                   loss_graph=tensorrec.loss_graphs.WMRBLossGraph())

# Fit the collaborative filter model
print("Training hybrid recommender")
hybrid_model.fit(interactions=sparse_train_ratings_4plus,
                 user_features=user_indicator_features,
                 item_features=full_item_features,
                 n_sampled_items=int(n_items * .01),
                 epochs=300,
                 verbose=True)


# This method consumes item ranks for each user and prints out recall@10 train/test metrics
def check_results(ranks):
    train_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=sparse_train_ratings_4plus,
        predicted_ranks=ranks,
        k=10
    ).mean()
    test_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=sparse_test_ratings_4plus,
        predicted_ranks=ranks,
        k=10
    ).mean()
    print("=== Recall at 10: Train: {:.4f} Test: {:.4f} ===".format(train_recall_at_10 * 100,
                                                                    test_recall_at_10 * 100))


# Check the results of the MF CF model
print("Hybrid model:")
predicted_ranks = hybrid_model.predict_rank(user_features=user_indicator_features,
                                            item_features=full_item_features)
check_results(predicted_ranks)

# Pull user 432's features out of the user features matrix and predict movie ranks for just that user
u432_features = sparse.csr_matrix(user_indicator_features)[432]
u432_rankings = hybrid_model.predict_rank(user_features=u432_features,
                                          item_features=full_item_features)[0]

# Get internal IDs of User 432's top 10 recommendations
# These are sorted by item ID, not by rank
# This may contain items with which User 432 has already interacted
u432_top_ten_recs = numpy.where(u432_rankings <= 10)[0]
print("User 432 recommendations:")
for m in u432_top_ten_recs:
    print(movie_titles_by_internal_id[m])
