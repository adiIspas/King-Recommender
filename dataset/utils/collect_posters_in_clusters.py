import os
import pandas as pd
import csv
import shutil

dataset = '../../../king-rec-dataset/ml-latest-small/images/'
number_of_clusters = 18
model = 'resnet50'
clusters_dir = '../../../king-rec-dataset/ml-latest-small/results/clusters/models/' + model + '/' + str(number_of_clusters) + '/'
# clusters_dir = './clusters/models/' + model + '/' + str(number_of_clusters) + '/'


def collect_posters():
    data = pd.read_csv('./posters_clusters/all_movies_1_poster_clusters_' + model + '.csv')

    # create directories
    for idx in range(1, number_of_clusters + 1):
        os.makedirs(clusters_dir + str(idx), exist_ok=True)

    # move posters into associated cluster
    for index, row in data.iterrows():
        src = dataset + str(int(row['0'])) + '/posters/' + str(int(row['1'])) + '.jpg'
        dest = clusters_dir + str(int(row['cluster_' + str(number_of_clusters)]) + 1) + '/' + str(int(row['0'])) + '_' + str(int(row['1'])) + '.jpg'

        if os.path.isfile(src):
            shutil.copy(src, dest)

    print('Done')


collect_posters()

dataset2 = '../../../king-rec-dataset/ml-latest-small/'


def get_items_ids():
    item_ids = set()

    with open(dataset2 + 'movies.csv', 'r') as movies_file:
        reader = csv.reader(movies_file, delimiter=',')
        next(reader)  # skip header

        for row in reader:
            item_ids.add(int(row[0]))

    return item_ids


def count_movies():
    movies = get_items_ids()

    idx = 1
    for item in movies:
        print(idx, item)
        idx = idx + 1


# count_movies()
