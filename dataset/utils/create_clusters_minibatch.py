import csv
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans


dataset = '../../../king-rec-dataset/ml-latest-small/'


def get_items_ids():
    item_ids = set()

    with open(dataset + 'movies.csv', 'r') as movies_file:
        reader = csv.reader(movies_file, delimiter=',')
        next(reader)  # skip header

        for row in reader:
            item_ids.add(int(row[0]))

    return item_ids


def explore_clusters():
    batch_size = 7

    models = ['vgg16', 'vgg19', 'inception_v3', 'resnet50', 'NASNet']

    for model in models:
        csv_path = './posters_features/sanity-check/' + model + '-sanity-check.csv'

        movie_poster_clusters = pd.DataFrame()
        for n_clusters in [7]:
            final_clusters = pd.Series()
            print('Process cluster', n_clusters)

            k_means = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_iter=300, tol=0.0001)

            reader_chunks = pd.read_csv(csv_path, delimiter=',', header=None, chunksize=batch_size)
            for chunk in reader_chunks:
                print('Processing chunk ...')

                feature_list = pd.DataFrame(data=chunk)

                movie_poster_clusters = movie_poster_clusters.append(feature_list.iloc[:, :2])

                feature_list = feature_list.iloc[:, 2:]
                feature_list_np = np.array(feature_list)

                k_means.partial_fit(feature_list_np)

            reader_chunks = pd.read_csv(csv_path, delimiter=',', header=None, chunksize=batch_size)
            for chunk in reader_chunks:
                print('Predicting chunk ...')

                feature_list = pd.DataFrame(data=chunk)

                feature_list = feature_list.iloc[:, 2:]
                feature_list_np = np.array(feature_list)

                labels = k_means.predict(feature_list_np)

                final_clusters = final_clusters.append(pd.Series(labels))

            name = model

            cluster_name = 'cluster_' + str(n_clusters)
            movie_poster_clusters[cluster_name] = pd.Series(final_clusters.values, index=movie_poster_clusters.index)

        movie_poster_clusters.to_csv('movies_1_poster_clusters_' + name + '.csv')


def main():
    explore_clusters()


if __name__ == "__main__":
    main()
