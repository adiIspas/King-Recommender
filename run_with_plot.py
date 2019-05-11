from kingrec import KingRec
from kingrec.evaluation import precision_at_k
from kingrec.evaluation import recall_at_k
from kingrec.evaluation import auc_score
from kingrec.dataset import init_movielens
import matplotlib.pyplot as plt
import numpy as np


k = 10
threads = 16
final_test_auc = []
final_test_precision = []
dataset = '../king-rec-dataset/ml-latest-small'


def load_auc_params(optimized_for=None):
    if optimized_for is None:
        print('Optimized for non features')

        optimal_epochs = 300
        optimal_learning_rate = 0.013125743984880447
        optimal_no_components = 169
        optimal_alpha = 2.6154143367150727e-06
        optimal_scaling = 0.04382333041868763

        return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

    if optimized_for == 'genres':
        print('Optimized for genres')

        optimal_epochs = 300
        optimal_learning_rate = 0.026238747910509397
        optimal_no_components = 193
        optimal_alpha = 0.0027085249085071626
        optimal_scaling = 0.07322973067589604

        return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

    if optimized_for == 'clusters':
        print('Optimized for clusters')

        optimal_epochs = 300
        optimal_learning_rate = 0.0570326091236193
        optimal_no_components = 68
        optimal_alpha = 0.0029503539747277366
        optimal_scaling = 0.02563602355611453

        return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

    if optimized_for == 'genres_clusters':
        print('Optimized for genres and clusters')

        optimal_epochs = 300
        optimal_learning_rate = 0.027730397776550147
        optimal_no_components = 189
        optimal_alpha = 0.0011133373244076297
        optimal_scaling = 0.4922360335772573

        return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling


for features in [None, ['genres'], ['clusters'], ['genres', 'clusters']]:
    movielens = init_movielens(dataset, min_rating=3.5, k=k, item_features=features)

    if features is None:
        epochs, learning_rate, no_components, alpha, scaling = load_auc_params(optimized_for=features)
    elif len(features) == 1:
        epochs, learning_rate, no_components, alpha, scaling = load_auc_params(optimized_for=features[0])
    elif len(features) == 2:
        epochs, learning_rate, no_components, alpha, scaling = load_auc_params(optimized_for=features[0] + '_' + features[1])

    train = movielens['train']
    test = movielens['test']
    item_features = movielens['item_features']

    king_rec = KingRec(no_components=no_components, learning_rate=learning_rate, alpha=alpha, scale=scaling, loss='warp')
    model = king_rec.model

    train_auc_scores = []
    test_auc_scores = []

    train_precision_scores = []
    test_precision_scores = []

    train_recall_scores = []
    test_recall_scores = []

    for epoch in range(epochs):
        print('Epoch:', epoch)
        model.fit_partial(train, item_features=item_features, epochs=1)

        train_precision = precision_at_k(model, train, item_features=item_features, k=k, num_threads=threads).mean()
        test_precision = precision_at_k(model, test, item_features=item_features, k=k, num_threads=threads).mean()

        train_recall = recall_at_k(model, train, item_features=item_features, k=k, num_threads=threads).mean()
        test_recall = recall_at_k(model, test, item_features=item_features, k=k, num_threads=threads).mean()

        train_auc = auc_score(model, train, item_features=item_features, num_threads=threads).mean()
        test_auc = auc_score(model, test, item_features=item_features, num_threads=threads).mean()

        train_auc_scores.append(train_auc)
        test_auc_scores.append(test_auc)

        train_precision_scores.append(train_precision)
        test_precision_scores.append(test_precision)

        train_recall_scores.append(train_recall)
        test_recall_scores.append(test_recall)

    final_test_auc.append(test_auc_scores)
    final_test_precision.append(test_precision_scores)

# plot results
plt.figure()
for auc_scores in final_test_auc:
    x = np.arange(len(auc_scores))

    max_value = max(auc_scores)
    max_index = auc_scores.index(max_value)

    plt.plot(x, auc_scores, '-D', markevery=[max_index])

plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Accuracy per model')
plt.legend(['without features: ' + str(np.round(max(final_test_auc[0]) * 100, decimals=2)) + '%',
            'genres: ' + str(np.round(max(final_test_auc[1]) * 100, decimals=2)) + '%',
            'clusters: ' + str(np.round(max(final_test_auc[2]) * 100, decimals=2)) + '%',
            'genres + clusters: ' + str(np.round(max(final_test_auc[3]) * 100, decimals=2)) + '%'
            ])
plt.savefig('auc-comparison.jpg')
plt.show()

plt.figure()
for precision_scores in final_test_precision:
    x = np.arange(len(precision_scores))

    max_value = max(precision_scores)
    max_index = precision_scores.index(max_value)

    plt.plot(x, precision_scores, '-D', markevery=[max_index])

plt.ylabel('Precision')
plt.xlabel('Epochs')
plt.title('Precision per model')
plt.legend(['without features: ' + str(np.round(max(final_test_precision[0]) * 100, decimals=2)) + '%',
            'genres: ' + str(np.round(max(final_test_precision[1]) * 100, decimals=2)) + '%',
            'clusters: ' + str(np.round(max(final_test_precision[2]) * 100, decimals=2)) + '%',
            'genres + clusters: ' + str(np.round(max(final_test_precision[3]) * 100, decimals=2)) + '%'
            ])
plt.savefig('precision-comparison.jpg')
plt.show()
