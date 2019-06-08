from kingrec import KingRec
from kingrec.evaluation import precision_at_k
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

        optimal_epochs = 100
        optimal_learning_rate = 0.0570326091236193
        optimal_no_components = 68
        optimal_alpha = 0.0029503539747277366
        optimal_scaling = 0.02563602355611453

        return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

    if optimized_for == 'genres_clusters':
        print('Optimized for genres and clusters')

        optimal_epochs = 200
        optimal_learning_rate = 0.027730397776550147
        optimal_no_components = 189
        optimal_alpha = 0.0011133373244076297
        optimal_scaling = 0.4922360335772573

        return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling


def load_precision_params(optimized_for=None):
    if optimized_for is None:
        print('Optimized for non features')

        optimal_epochs = 141
        optimal_learning_rate = 0.043040683676705736
        optimal_no_components = 21
        optimal_alpha = 0.00541554967720231
        optimal_scaling = 0.014726505321746962

        return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

    if optimized_for == 'genres':
        print('Optimized for genres')

        optimal_epochs = 136
        optimal_learning_rate = 0.075490395178898
        optimal_no_components = 82
        optimal_alpha = 0.007065549151367718
        optimal_scaling = 0.00799962475267643

        return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

    if optimized_for == 'clusters':
        print('Optimized for clusters')

        optimal_epochs = 63
        optimal_learning_rate = 0.05647434188275842
        optimal_no_components = 98
        optimal_alpha = 0.0031993742820159436
        optimal_scaling = 0.0933642796909375

        return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

    if optimized_for == 'genres_clusters':
        print('Optimized for genres and clusters')

        optimal_epochs = 96
        optimal_learning_rate = 0.1703221223672566
        optimal_no_components = 22
        optimal_alpha = 0.004206346506337412
        optimal_scaling = 0.041303781930858034

        return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling


def load_precision_params_clusters(optimized_for='clusters', model='vgg19'):
    print('Model:', model)

    if model == 'vgg19':
        if optimized_for == 'clusters':
            print('Optimized for clusters')

            optimal_epochs = 232
            optimal_learning_rate = 0.07171978672352887
            optimal_no_components = 42
            optimal_alpha = 0.006517845577815826
            optimal_scaling = 0.016142300018137722

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

        if optimized_for == 'genres_clusters':
            print('Optimized for genres and clusters')

            optimal_epochs = 218
            optimal_learning_rate = 0.12470857345083873
            optimal_no_components = 73
            optimal_alpha = 0.005478316990150038
            optimal_scaling = 0.04637764141484815

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling
    elif model == 'inception_v3':
        if optimized_for == 'clusters':
            print('Optimized for clusters')

            optimal_epochs = 119
            optimal_learning_rate = 0.00852211930222011
            optimal_no_components = 192
            optimal_alpha = 7.276515301192984e-05
            optimal_scaling = 0.027052254503857717

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

        if optimized_for == 'genres_clusters':
            print('Optimized for genres and clusters')

            optimal_epochs = 245
            optimal_learning_rate = 0.028963892665938032
            optimal_no_components = 43
            optimal_alpha = 0.0006238083410955659
            optimal_scaling = 0.36579038826022736

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling
    elif model == 'resnet50':
        if optimized_for == 'clusters':
            print('Optimized for clusters')

            optimal_epochs = 88
            optimal_learning_rate = 0.07492160698420884
            optimal_no_components = 21
            optimal_alpha = 0.004634987385145838
            optimal_scaling = 0.028198967823831238

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

        if optimized_for == 'genres_clusters':
            print('Optimized for genres and clusters')

            optimal_epochs = 224
            optimal_learning_rate = 0.04214027912721876
            optimal_no_components = 186
            optimal_alpha = 0.008676073688466915
            optimal_scaling = 0.0024915458462563605

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling


def load_auc_params_clusters(optimized_for='clusters', model='vgg19'):
    print('Model:', model)

    if model == 'vgg19':
        if optimized_for == 'clusters':
            print('Optimized for clusters')

            optimal_epochs = 89
            optimal_learning_rate = 0.018841927704689492
            optimal_no_components = 139
            optimal_alpha = 0.0008662511914237855
            optimal_scaling = 0.2864763834214625

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

        if optimized_for == 'genres_clusters':
            print('Optimized for genres and clusters')

            optimal_epochs = 236
            optimal_learning_rate = 0.031860755009764305
            optimal_no_components = 139
            optimal_alpha = 0.0010930770083784052
            optimal_scaling = 0.8362665749306415

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling
    elif model == 'inception_v3':
        if optimized_for == 'clusters':
            print('Optimized for clusters')

            optimal_epochs = 232
            optimal_learning_rate = 0.02981041359364386
            optimal_no_components = 84
            optimal_alpha = 0.004287524090264805
            optimal_scaling = 0.040501994149651166

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

        if optimized_for == 'genres_clusters':
            print('Optimized for genres and clusters')

            optimal_epochs = 250
            optimal_learning_rate = 0.019411170816577752
            optimal_no_components = 136
            optimal_alpha = 0.0008323333176050233
            optimal_scaling = 0.4767783602102349

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling
    elif model == 'resnet50':
        if optimized_for == 'clusters':
            print('Optimized for clusters')

            optimal_epochs = 198
            optimal_learning_rate = 0.016780379637566917
            optimal_no_components = 169
            optimal_alpha = 0.0012939223653296507
            optimal_scaling = 0.6692069103186539

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling

        if optimized_for == 'genres_clusters':
            print('Optimized for genres and clusters')

            optimal_epochs = 211
            optimal_learning_rate = 0.09767064566975311
            optimal_no_components = 48
            optimal_alpha = 0.003428832598553235
            optimal_scaling = 0.11239835090728653

            return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling


def general_models(params_to_load=load_auc_params):
    for features in [None, ['genres'], ['clusters'], ['genres', 'clusters']]:
        movielens = init_movielens(dataset, min_rating=3.5, k=k, item_features=features)

        if features is None:
            epochs, learning_rate, no_components, alpha, scaling = params_to_load(optimized_for=features)
        elif len(features) == 1:
            epochs, learning_rate, no_components, alpha, scaling = params_to_load(optimized_for=features[0])
        elif len(features) == 2:
            epochs, learning_rate, no_components, alpha, scaling = params_to_load(optimized_for=features[0] + '_' + features[1])

        train = movielens['train']
        test = movielens['test']
        item_features = movielens['item_features']

        king_rec = KingRec(no_components=no_components, learning_rate=learning_rate, alpha=alpha, scale=scaling,
                           loss='warp')
        model = king_rec.model

        train_auc_scores = []
        test_auc_scores = []

        train_precision_scores = []
        test_precision_scores = []

        for epoch in range(epochs):
            print('Epoch:', epoch)
            model.fit_partial(train, item_features=item_features, epochs=1)

            train_precision = precision_at_k(model, train, item_features=item_features, k=k, num_threads=threads).mean()
            test_precision = precision_at_k(model, test, item_features=item_features, k=k, num_threads=threads).mean()

            train_auc = auc_score(model, train, item_features=item_features, num_threads=threads).mean()
            test_auc = auc_score(model, test, item_features=item_features, num_threads=threads).mean()

            train_auc_scores.append(train_auc)
            test_auc_scores.append(test_auc)

            train_precision_scores.append(train_precision)
            test_precision_scores.append(test_precision)

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


def clusters_models(params_to_load_clusters=load_auc_params_clusters):
    for model in ['vgg19', 'inception_v3', 'resnet50']:
        # features = ['clusters']
        features = ['genres', 'clusters']
        movielens = init_movielens(dataset, min_rating=3.5, k=k, item_features=features, model=model)

        if len(features) == 1:
            epochs, learning_rate, no_components, alpha, scaling = params_to_load_clusters(optimized_for=features[0], model=model)
        elif len(features) == 2:
            epochs, learning_rate, no_components, alpha, scaling = params_to_load_clusters(optimized_for=features[0] + '_' + features[1], model=model)

        train = movielens['train']
        test = movielens['test']
        item_features = movielens['item_features']

        king_rec = KingRec(no_components=no_components, learning_rate=learning_rate, alpha=alpha, scale=scaling, loss='warp')
        model = king_rec.model

        train_auc_scores = []
        test_auc_scores = []

        train_precision_scores = []
        test_precision_scores = []

        for epoch in range(epochs):
            print('Epoch:', epoch)
            model.fit_partial(train, item_features=item_features, epochs=1)

            train_precision = precision_at_k(model, train, item_features=item_features, k=k, num_threads=threads).mean()
            test_precision = precision_at_k(model, test, item_features=item_features, k=k, num_threads=threads).mean()

            train_precision_scores.append(train_precision)
            test_precision_scores.append(test_precision)

            # train_auc = auc_score(model, train, item_features=item_features, num_threads=threads).mean()
            # test_auc = auc_score(model, test, item_features=item_features, num_threads=threads).mean()

            # train_auc_scores.append(train_auc)
            # test_auc_scores.append(test_auc)

        # final_test_auc.append(test_auc_scores)
        final_test_precision.append(test_precision_scores)

    # plot results
    plt.figure()
    for auc_scores in final_test_auc:
        x = np.arange(len(auc_scores))

        max_value = max(auc_scores)
        max_index = auc_scores.index(max_value)

        # plt.plot(x, auc_scores, '-D', markevery=[max_index])
        plt.plot(x, auc_scores)

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('Accuracy per model with genres and clusters metadata')
    plt.legend(['vgg19: ' + str(np.round(max(final_test_auc[0]) * 100, decimals=2)) + '%',
                'inception_v3: ' + str(np.round(max(final_test_auc[1]) * 100, decimals=2)) + '%',
                'resnet50: ' + str(np.round(max(final_test_auc[2]) * 100, decimals=2)) + '%',
                ])
    plt.savefig('auc-comparison.jpg')
    plt.show()

    plt.figure()
    for precision_scores in final_test_precision:
        x = np.arange(len(precision_scores))

        max_value = max(precision_scores)
        max_index = precision_scores.index(max_value)

        # plt.plot(x, precision_scores, '-D', markevery=[max_index])
        plt.plot(x, precision_scores)

    plt.ylabel('Precision')
    plt.xlabel('Epochs')
    plt.title('Precision per model with genres and clusters metadata')
    plt.legend(['vgg19: ' + str(np.round(max(final_test_precision[0]) * 100, decimals=2)) + '%',
                'inception_v3: ' + str(np.round(max(final_test_precision[1]) * 100, decimals=2)) + '%',
                'resnet50: ' + str(np.round(max(final_test_precision[2]) * 100, decimals=2)) + '%',
                ])
    plt.savefig('precision-comparison.jpg')
    plt.show()


general_models(load_precision_params)

# clusters_models(load_precision_params_clusters)
