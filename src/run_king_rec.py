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
dataset_path = '../data/raw/ml-latest-small/dataset/'

# create model and add useful information
recommender = KingRecommender(dataset_path)

# define dataset keys
ratings = 'ratings'
movies = 'movies'

# add dataset file paths for each file
recommender.add_dataset_file_path(ratings, file_ratings)
recommender.add_dataset_file_path(movies, file_movies)

# create internal ids for users and items
ratings_file = recommender.get_dataset_file_path(ratings)
internal_user_ids, internal_item_ids = create_internal_ids(ratings_file)

recommender.add_internal_user_ids(internal_user_ids)
recommender.add_internal_item_ids(internal_item_ids)

# read and add dataset to model
raw_ratings = read_ratings(ratings_file, internal_user_ids, internal_item_ids)
recommender.add_dataset_entry(ratings, raw_ratings)

sparse_train_ratings_4plus, sparse_test_ratings_4plus = \
    create_test_train_interactions(raw_ratings, recommender.n_users, recommender.n_items)

recommender.add_interactions(sparse_train_ratings_4plus)
recommender.add_test_interactions(sparse_test_ratings_4plus)

# train_ratings, test_ratings = split_data(raw_ratings, train_size=0.8)
#
# sparse_train_ratings = interactions_list_to_sparse_matrix(train_ratings, n_users, n_items)
# sparse_test_ratings = interactions_list_to_sparse_matrix(test_ratings, n_users, n_items)
#
# sparse_train_ratings_4plus = sparse_train_ratings.multiply(sparse_train_ratings >= 4.0)
# sparse_test_ratings_4plus = sparse_test_ratings.multiply(sparse_test_ratings >= 4.0)

# Construct indicator features for users and items
# user_indicator_features = sparse.identity(n_users)
# item_indicator_features = sparse.identity(n_items)

movie_titles_by_internal_id, movie_genres_by_internal_id = \
    read_genres(recommender.get_dataset_file_path(movies), internal_item_ids=recommender.internal_items_ids)
movie_genre_features = create_genres_features(movie_genres_by_internal_id, recommender.n_items)

# Try concatenating the genres on to the indicator features for a hybrid recommender system
full_item_features = sparse.hstack([sparse.identity(recommender.n_items), movie_genre_features])

# Add features to the model
recommender.user_features = sparse.identity(recommender.n_users)
recommender.item_features = full_item_features

# Build a matrix factorization collaborative filter model
# hybrid_model = tensorrec.TensorRec(n_components=5,
#                                    loss_graph=tensorrec.loss_graphs.WMRBLossGraph())

# Fit the collaborative filter model
# print("Training hybrid recommender")
# hybrid_model.fit(interactions=sparse_train_ratings_4plus,
#                  user_features=recommender.user_features,
#                  item_features=recommender.item_features,
#                  n_sampled_items=int(recommender.n_items * .01),
#                  epochs=300,
#                  verbose=True)


# This method consumes item ranks for each user and prints out recall@10 train/test metrics
# def check_results(ranks):
#     train_recall_at_10 = tensorrec.eval.recall_at_k(
#         test_interactions=sparse_train_ratings_4plus,
#         predicted_ranks=ranks,
#         k=10
#     ).mean()
#     test_recall_at_10 = tensorrec.eval.recall_at_k(
#         test_interactions=sparse_test_ratings_4plus,
#         predicted_ranks=ranks,
#         k=10
#     ).mean()
#     print("=== Recall at 10: Train: {:.4f} Test: {:.4f} ===".format(train_recall_at_10 * 100,
#                                                                     test_recall_at_10 * 100))
#
#
# # Check the results of the MF CF model
# print("Hybrid model:")
# predicted_ranks = hybrid_model.predict_rank(user_features=user_indicator_features,
#                                             item_features=full_item_features)
# check_results(predicted_ranks)

recommender.fit(300)
recommender.check_results()

# === IS DONE === #

# # Pull user 432's features out of the user features matrix and predict movie ranks for just that user
# u432_features = sparse.csr_matrix(user_indicator_features)[432]
# u432_rankings = hybrid_model.predict_rank(user_features=u432_features,
#                                           item_features=full_item_features)[0]
#
# # Get internal IDs of User 432's top 10 recommendations
# # These are sorted by item ID, not by rank
# # This may contain items with which User 432 has already interacted
# u432_top_ten_recs = numpy.where(u432_rankings <= 10)[0]
# print("User 432 recommendations:")
# for m in u432_top_ten_recs:
#     print(movie_titles_by_internal_id[m])
