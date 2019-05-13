import os
import numpy as np
import csv
import pandas as pd

from keras.layers import Input
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetLarge
from keras.applications.imagenet_utils import preprocess_input

dataset = '../../../king-rec-dataset/ml-latest-small/'
base_path = 'images/'
# base_path = 'clusters_sanity_check/'
max_posters_per_movie = 1


def get_int(filename):
    return int(filename.split('.')[0])


def get_items_ids():
    item_ids = set()

    with open(dataset + 'movies.csv', 'r') as movies_file:
        reader = csv.reader(movies_file, delimiter=',')
        next(reader)  # skip header

        for row in reader:
            item_ids.add(int(row[0]))

    return item_ids


def extract_images_features():
    movies = list(get_items_ids())
    subdir = [dataset + base_path + str(movie) + '/posters/' for movie in movies]
    models = [
              VGG16(weights='imagenet', include_top=False),
              VGG19(weights='imagenet', include_top=False),
              InceptionV3(weights='imagenet', include_top=False),
              ResNet50(weights='imagenet', include_top=False),
              NASNetLarge(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
             ]
    total_movies = len(subdir)
    for current_movie, dirname in enumerate(subdir):
        movie_idx = int([s for s in dirname.split('/') if s.isdigit()][0])
        filenames = sorted(os.listdir(dirname), key=get_int)[0:max_posters_per_movie]

        for _, file_name in enumerate(filenames):
            poster_idx = int(file_name.split('.')[0])

            img = image.load_img(dirname + '/' + file_name, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)

            for model in models:
                feature = model.predict(img_data)
                feature_np = np.array(feature)
                feature = feature_np.flatten()

                data_to_save = np.append([movie_idx, poster_idx], feature)
                data = pd.DataFrame([data_to_save])
                data.to_csv(model.name + '-' + str(max_posters_per_movie) + '-posters' + '.csv',
                            mode='a', sep=',', index=False, header=False)

                print(str(current_movie + 1) + '/' + str(total_movies) + ':', 'movie id:', movie_idx, '  poster id:', poster_idx,
                      '  model name:', model.name, '  total features:', len(feature))


def main():
    extract_images_features()


if __name__ == "__main__":
    main()
