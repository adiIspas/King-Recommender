import os
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn import metrics
from keras.layers import Input
from sklearn.cluster import KMeans
from keras.preprocessing import image
from sklearn.decomposition import PCA
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetLarge
from keras.applications.imagenet_utils import preprocess_input

dataset = '../../../king-rec-dataset/ml-latest-small/'
pca_components = 5000
max_posters_per_movie = 3
max_movies_numbers = 5000
pca_reduction = True


def get_items_ids():
    item_ids = set()

    with open(dataset + 'movies.csv', 'r') as movies_file:
        reader = csv.reader(movies_file, delimiter=',')
        next(reader)  # skip header

        for row in reader:
            item_ids.add(int(row[0]))

    return item_ids


def get_pca_features(features):
    features = np.array([feature for feature in features])
    pca = PCA(n_components=pca_components, whiten=True)
    return pca.fit_transform(features)


def explore_clusters():
    movies = list(get_items_ids())[0:max_movies_numbers]
    subdir = [dataset + 'images/' + str(movie) + '/posters/' for movie in movies]
    models = [VGG16(weights='imagenet', include_top=False)]
        # , VGG19(weights='imagenet', include_top=False),
        #       InceptionV3(weights='imagenet', include_top=False), ResNet50(weights='imagenet', include_top=False),
        #       NASNetLarge(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))]

    clusters = range(2, 22, 2)
    models_results = dict()
    colors = ['r', 'y', 'b', 'g', 'c']

    images_data = []
    print('Start read posters')
    for idx, dirname in enumerate(subdir):
        filenames = os.listdir(dirname)[0:max_posters_per_movie]
        for i, file_name in enumerate(filenames):
            img = image.load_img(dirname + '/' + file_name, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            images_data.append(img_data)

    print('Total number of posters:', len(images_data))

    print('Start evaluate models')
    for model in models:
        feature_list = []

        for img_data in images_data:
            feature = model.predict(img_data)
            feature_np = np.array(feature)
            feature_list.append(feature_np.flatten())

        if pca_reduction:
            feature_list = get_pca_features(feature_list)

        for n_clusters in clusters:
            feature_list_np = np.array(feature_list)
            k_means = KMeans(n_clusters=n_clusters).fit(feature_list_np)

            name = model.name
            result = metrics.silhouette_score(feature_list_np, k_means.labels_)

            if name not in models_results:
                results = []
            else:
                results = models_results.pop(name)

            results.append(result)
            models_results.update({name: results})
            print('silhouette score on', name, 'with', n_clusters, 'clusters:', result)

    n_groups = len(list(clusters))
    index = np.arange(n_groups)
    bar_width = 0.15
    current_index = 0

    for key, values in models_results.items():
        plt.bar(index + bar_width * current_index, values, bar_width,
                color=colors[current_index],
                label=key)
        current_index += 1

    if pca_reduction:
        fig_name = "silhouette-" + str(max_movies_numbers) + "-" + str(max_posters_per_movie) + "-" + str(pca_components)
    else:
        fig_name = "silhouette-" + str(max_movies_numbers) + "-" + str(max_posters_per_movie)

    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score by model')
    plt.xticks(index + bar_width, list(clusters))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name + '.jpg')
    plt.show()


def generate_posters_clusters():
    movies = get_items_ids()


def main():
    explore_clusters()


if __name__ == "__main__":
    main()
