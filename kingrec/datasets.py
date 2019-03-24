import csv
from lightfm.data import Dataset
from sklearn.model_selection import train_test_split

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
        ratings_train, ratings_test = train_test_split(ratings, test_size=0.3)

        train_dataset.fit(users=users, items=items)
        test_dataset.fit(users=users, items=items)

        train_interactions = train_dataset.build_interactions(ratings_train)[1]
        test_interactions = test_dataset.build_interactions(ratings_test)[1]

        data.update({'train': train_interactions})
        data.update({'test': test_interactions})

        return data
