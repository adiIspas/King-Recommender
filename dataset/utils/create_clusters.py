import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans


dataset = '../../../king-rec-dataset/ml-latest-small/'
type_of_features = '1000-movies'
# type_of_features = 'sanity-check'


def get_items_ids():
    item_ids = set()

    with open(dataset + 'movies.csv', 'r') as movies_file:
        reader = csv.reader(movies_file, delimiter=',')
        next(reader)  # skip header

        for row in reader:
            item_ids.add(int(row[0]))

    return item_ids


def explore_clusters():
    clusters = range(2, 22, 2)
    models_results = dict()
    colors = ['r', 'y', 'b', 'g', 'c']

    models = ['vgg16', 'vgg19', 'inception_v3', 'resnet50', 'NASNet']

    for model in models:
        print('Reading data ...')
        feature_list = np.loadtxt('./posters_features/' + type_of_features + '/' + model + '-sanity-check.csv', delimiter=',')
        print('Complete read data.')

        movie_poster_clusters = pd.DataFrame(feature_list[:, :2])

        feature_list = feature_list[:, 2:]
        feature_list_np = np.array(feature_list)
        for n_clusters in clusters:
            k_means = KMeans(n_clusters=n_clusters).fit(feature_list_np)

            name = model
            result = metrics.silhouette_score(feature_list_np, k_means.labels_)

            if name not in models_results:
                results = []
            else:
                results = models_results.pop(name)

            cluster_name = 'cluster_' + str(n_clusters)
            movie_poster_clusters[cluster_name] = pd.Series(k_means.labels_)

            results.append(result)
            models_results.update({name: results})
            print('silhouette score on', name, 'with', n_clusters, 'clusters:', result)

        movie_poster_clusters.to_csv('movies_1_poster_clusters_' + name + '.csv')

    n_groups = len(list(clusters))
    index = np.arange(n_groups)
    bar_width = 0.15
    current_index = 0

    for key, values in models_results.items():
        plt.bar(index + bar_width * current_index, values, bar_width,
                color=colors[current_index],
                label=key)
        current_index += 1

    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score by model')
    plt.xticks(index + bar_width, list(clusters))
    plt.legend()
    plt.tight_layout()
    plt.savefig('silhouette-score.jpg')
    plt.show()


def main():
    explore_clusters()


if __name__ == "__main__":
    main()
