import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

dataset = '../../../king-rec-dataset/ml-latest-small/'
# pca_components = 5000
# max_posters_per_movie = 3
# max_movies_numbers = 5000
# pca_reduction = True
# n_rows = 300


def get_items_ids():
    item_ids = set()

    with open(dataset + 'movies.csv', 'r') as movies_file:
        reader = csv.reader(movies_file, delimiter=',')
        next(reader)  # skip header

        for row in reader:
            item_ids.add(int(row[0]))

    return item_ids


# def get_pca_features(features):
#     features = np.array([feature for feature in features])
#     pca = PCA(n_components=pca_components, whiten=True)
#     return pca.fit_transform(features)


def explore_clusters():
    clusters = range(2, 8, 1)
    models_results = dict()
    colors = ['r', 'y', 'b', 'g', 'c']

    models = ['vgg16', 'vgg19', 'inception_v3', 'resnet50', 'NASNet']

    for model in models:
        print('Reading data ...')
        feature_list = np.loadtxt('./' + model + '-100-posters.csv', delimiter=',')
        # feature_list = feature_list[feature_list[:, 1] == 1]  # select just one poster per movie
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

        movie_poster_clusters.to_csv('sanity_check_movies_1_poster_clusters_' + name + '.csv')

    n_groups = len(list(clusters))
    index = np.arange(n_groups)
    bar_width = 0.15
    current_index = 0

    for key, values in models_results.items():
        plt.bar(index + bar_width * current_index, values, bar_width,
                color=colors[current_index],
                label=key)
        current_index += 1

    # if pca_reduction:
    #     fig_name = "silhouette-" + str(max_movies_numbers) + "-" + str(max_posters_per_movie) + "-" + str(pca_components)
    # else:
    #     fig_name = "silhouette-" + str(max_movies_numbers) + "-" + str(max_posters_per_movie)

    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score by model')
    plt.xticks(index + bar_width, list(clusters))
    plt.legend()
    plt.tight_layout()
    plt.savefig('silhouette-sanity-check.jpg')
    plt.show()


def generate_posters_clusters():
    movies = get_items_ids()


def main():
    explore_clusters()


if __name__ == "__main__":
    main()
