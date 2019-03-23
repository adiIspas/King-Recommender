import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input

models = [VGG16(weights='imagenet', include_top=False), VGG19(weights='imagenet', include_top=False),
          InceptionV3(weights='imagenet', include_top=False), ResNet50(weights='imagenet', include_top=False)]

clusters = range(2, 20, 2)
models_results = dict()
colors = ['r', 'y', 'b', 'g']


class ModelResult(object):
    def __init__(self, model_name, model_result):
        self.name = model_name
        self.result = model_result


for model in models:
    feature_list = []

    subdir = ['captain_marvel', 'triple_frontier', 'the_notebook', 'inception']
    for idx, dirname in enumerate(subdir):

        filenames = os.listdir(dirname)
        for i, file_name in enumerate(filenames):
            img = image.load_img(dirname + '/' + file_name, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)

            feature = model.predict(img_data)
            feature_np = np.array(feature)
            feature_list.append(feature_np.flatten())

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
fig = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
current_index = 0

for key, values in models_results.items():
    rects1 = plt.bar(index + bar_width * current_index, values, bar_width,
                     color=colors[current_index],
                     label=key)
    current_index += 1

plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score by model')
plt.xticks(index + bar_width, list(clusters))
plt.legend()

plt.tight_layout()
plt.show()
