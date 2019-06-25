from kingrec import KingRec
from kingrec.dataset import init_movielens


class InitModel(object):
    def __init__(self):
        self.k = 5
        self.threads = 16
        self.min_rating = 3.5
        self.optimal_epochs = 100
        self.optimal_no_components = 68
        self.optimal_alpha = 0.0029503539747277366
        self.optimal_scaling = 0.02563602355611453
        self.optimal_learning_rate = 0.0570326091236193
        self.dataset = '../king-rec-dataset/ml-latest-small'
        self.movielens = None
        self.model = None

    def load_model(self):
        self.movielens = init_movielens(self.dataset,
                                        min_rating=self.min_rating,
                                        k=self.k,
                                        item_features=['clusters'],
                                        test_percentage=0.01)

        train = self.movielens['train']
        item_features = self.movielens['item_features']

        king_rec = KingRec(no_components=self.optimal_no_components,
                           learning_rate=self.optimal_learning_rate,
                           alpha=self.optimal_alpha,
                           scale=self.optimal_scaling,
                           loss='warp')

        self.model = king_rec.model
        self.model.fit_partial(interactions=train, item_features=item_features,
                               epochs=self.optimal_epochs, verbose=True, num_threads=self.threads)

        return self.model

    def update_model(self, user_id, movie_id_index):
        pass
