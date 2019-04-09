import csv
from scipy import sparse
from lightfm.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

__all__ = ['init_movielens']


def init_movielens(path):
    train_dataset = Dataset()
    test_dataset = Dataset()
    data = dict()

    with open(path + '/ratings.csv', 'r') as ratings_file:
        reader = csv.reader(ratings_file, delimiter=',', )
        next(reader)  # skip header

        ratings = []
        users = set()
        items = set()
        for row in reader:
            users.add(int(row[0]))
            items.add(int(row[1]))
            ratings.append((int(row[0]), int(row[1]), float(row[2])))

        users = list(users)
        items = list(items)
        ratings_train, ratings_test = train_test_split(ratings, test_size=0.2)

        train_dataset.fit(users=users, items=items)
        test_dataset.fit(users=users, items=items)

        train_interactions = train_dataset.build_interactions(ratings_train)[1]
        test_interactions = test_dataset.build_interactions(ratings_test)[1]

        data.update({'train': train_interactions})
        data.update({'test': test_interactions})

    movie_genres, genres = __init_movies_genres(path)
    one_hot = MultiLabelBinarizer()

    movie_genre_features = one_hot.fit_transform(movie_genres.values())
    movie_genre_features = sparse.coo_matrix(movie_genre_features)

    item_indicator_features = sparse.identity(9742)  # TODO: trebuie fixata dimensiunea, pare ca lipsesc filme
    item_features = sparse.hstack([item_indicator_features, movie_genre_features])

    data.update({'item_features': item_features})

    return data


def __init_movies_genres(path):
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


def __init_movies_posters_clusters(path):
    pass
