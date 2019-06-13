from kingrec import KingRec
from kingrec.dataset import init_movielens


def load_params():
    # parameters for best model on accuracy
    optimal_epochs = 100
    optimal_learning_rate = 0.0570326091236193
    optimal_no_components = 68
    optimal_alpha = 0.0029503539747277366
    optimal_scaling = 0.02563602355611453

    return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling


def load_model():
    k = 5
    threads = 16
    dataset = '../king-rec-dataset/ml-latest-small'
    epochs, learning_rate, no_components, alpha, scaling = load_params()

    movielens = init_movielens(dataset, min_rating=3.5, k=k, item_features=['clusters'], test_percentage=0.0)

    train = movielens['train']
    item_features = movielens['item_features']

    king_rec = KingRec(no_components=no_components, learning_rate=learning_rate, alpha=alpha, scale=scaling,
                       loss='warp')

    model = king_rec.model
    model.fit_partial(interactions=train, item_features=item_features, epochs=epochs, verbose=True, num_threads=threads)

    return model


def update_model(model):
    pass
