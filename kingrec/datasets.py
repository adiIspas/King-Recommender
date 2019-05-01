import csv
import pandas as pd
import numpy as np
from scipy import sparse
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split

__all__ = ['init_movielens']


def init_movielens(path, min_rating=0.0, k=3, item_features=None, cluster_n=18):
    valid_item_features = {'genres': 'genres', 'clusters': 'clusters'}
    if item_features is not None:
        assert all(item in valid_item_features.values() for item in item_features), \
            'Your specified item features is invalid. You have to use one or more of this: ' \
            + ', '.join(valid_item_features)

    train_dataset = Dataset()
    test_dataset = Dataset()

    data = dict()
    min_interactions = dict()

    with open(path + '/ratings.csv', 'r') as ratings_file:
        reader = csv.reader(ratings_file, delimiter=',', )
        next(reader)  # skip header

        ratings = []
        users = set()
        items = set()
        for row in reader:
            user_id = int(row[0])
            item_id = int(row[1])

            users.add(user_id)
            items.add(item_id)

            rating = float(row[2])

            if rating >= min_rating:
                ratings.append((user_id, item_id, rating))
                __add_interaction(min_interactions, user_id)

        __info_no_of_min_interactions(k, 'No of interactions per user overall ==> ', min_interactions)

        users = list(users)
        items = list(items)

        users_column, items_column, ratings_column = zip(*ratings)
        ratings = sparse.coo_matrix((ratings_column, (users_column, items_column)))

        ratings_train, ratings_test = random_train_test_split(ratings, test_percentage=0.2,
                                                              random_state=np.random.RandomState(7))

        ratings_train_to_count = zip(ratings_train.row, ratings_train.col, ratings_train.data)
        ratings_train = zip(ratings_train.row, ratings_train.col, ratings_train.data)

        ratings_test_to_count = zip(ratings_test.row, ratings_test.col, ratings_test.data)
        ratings_test = zip(ratings_test.row, ratings_test.col, ratings_test.data)

        min_interactions = __count_train_test_min_interactions(ratings_train_to_count)
        __info_no_of_min_interactions(k, 'No of interactions per user on train ==> ', min_interactions)

        min_interactions = __count_train_test_min_interactions(ratings_test_to_count)
        __info_no_of_min_interactions(k, 'No of interactions per user on test ==> ', min_interactions)

        train_dataset.fit(users=users, items=items)
        test_dataset.fit(users=users, items=items)

        (train_interactions, train_weights) = train_dataset.build_interactions(ratings_train)
        (test_interactions, test_weights) = test_dataset.build_interactions(ratings_test)

        data.update({'train': train_interactions})
        data.update({'test': test_interactions})

    # add item features
    if item_features is not None:
        aggregated_features = []

        if valid_item_features.get('genres') in item_features:
            movie_genres, genres = __init_movies_genres(path)
            aggregated_features.append(movie_genres)

            train_dataset.fit_partial(item_features=genres)
            test_dataset.fit_partial(item_features=genres)

            train_dataset.fit_partial(items=list(movie_genres.keys()))
            test_dataset.fit_partial(items=list(movie_genres.keys()))

        if valid_item_features.get('clusters') in item_features:
            movies_posters_clusters, clusters = __init_movies_posters_clusters(path, cluster_n)
            aggregated_features.append(movies_posters_clusters)

            train_dataset.fit_partial(item_features=clusters)
            test_dataset.fit_partial(item_features=clusters)

            train_dataset.fit_partial(items=list(movies_posters_clusters.keys()))
            test_dataset.fit_partial(items=list(movies_posters_clusters.keys()))

        aggregated_features = __aggregate_features(aggregated_features)
        item_features = train_dataset.build_item_features(((movie_id, aggregated_features.get(movie_id))
                                                           for movie_id in aggregated_features.keys()))

        _ = test_dataset.build_item_features(((movie_id, aggregated_features.get(movie_id))
                                              for movie_id in aggregated_features.keys()))

        data.update({'item_features': item_features})
    else:
        data.update({'item_features': None})

    return data


def __init_movies_genres(path):
    print('Init movies genres ...')
    movies_genres = dict()
    genres = set()

    with open(path + '/movies.csv', 'r') as ratings_file:
        reader = csv.reader(ratings_file, delimiter=',', )
        next(reader)  # skip header

        for row in reader:
            movies_genres.update({int(row[0]): str(row[2]).split('|')})

            for genre in str(row[2]).split('|'):
                genres.add(genre)

    return movies_genres, genres


def __init_movies_posters_clusters(path, cluster_n):
    print('Init movies posters clusters ...')

    movie_clusters = dict()
    movies_posters_clusters = pd.read_csv(path + '/movies_1_poster_clusters_vgg19.csv')

    for index, _ in movies_posters_clusters.iterrows():
        movie_id = int(movies_posters_clusters['0'][index])
        poster_cluster = movies_posters_clusters['cluster_' + str(cluster_n)][index] + 1

        if movie_id not in movie_clusters:
            movie_clusters.update({movie_id: ['cluster_' + str(poster_cluster)]})
        else:
            movie_clusters.update({movie_id: movie_clusters.get(movie_id).append('cluster_' + str(poster_cluster))})

    return movie_clusters, ['cluster_' + str(idx + 1) for idx in range(cluster_n)]


def __aggregate_features(features_dicts):
    aggregated_features = dict()

    for features_dict in features_dicts:
        for item_id in features_dict.keys():
            if item_id not in aggregated_features:
                aggregated_features.update({item_id: features_dict.get(item_id)})
            else:
                aggregated_features.update({item_id: aggregated_features.get(item_id) + features_dict.get(item_id)})

    return aggregated_features


def __count_train_test_min_interactions(ratings_train_to_count):
    min_interactions = dict()
    for user_id, item_id, rating in ratings_train_to_count:
        __add_interaction(min_interactions, user_id)

    return min_interactions


def __add_interaction(min_interactions, user_id):
    if user_id not in min_interactions:
        min_interactions.update({user_id: 1})
    else:
        min_interactions.update({user_id: 1 + min_interactions.get(user_id)})


def __info_no_of_min_interactions(k, message, min_interactions):
    less_k = len([no for no in list(min_interactions.values()) if no < k])
    total = len(min_interactions.values())

    print(message, 'Min:', min(min_interactions.values()),
          'Mean:', np.mean(list(min_interactions.values())), 'Max:', max(min_interactions.values()),
          '<k=' + str(k) + ':', str(less_k) + '/' + str(total))
