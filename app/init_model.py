from kingrec import KingRec
from kingrec.dataset import init_movielens
from scipy import sparse


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
        self.train_mapping = None
        self.movielens = None
        self.model = None
        self.user_id = 1

    def load_model(self):
        self.movielens = init_movielens(self.dataset,
                                        min_rating=self.min_rating,
                                        k=self.k,
                                        item_features=['clusters'],
                                        test_percentage=0.01)

        self.train_mapping = self.movielens['train-mapping']

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

    def update_model(self, user_id, movie_id):
        user_id_mapping = self.train_mapping[0].get(user_id)
        movie_id_mapping = self.train_mapping[2].get(movie_id)

        user_id_index = self.movielens['train'].row[user_id_mapping]
        movie_id_index = self.movielens['train'].col[movie_id_mapping]
        data = self.movielens['train'].toarray()
        data[user_id_index][movie_id_index] = -1

        # data trebuie sa fie de aceasi dimensiune cu row si col
        shape = data.shape
        data = data.flatten()
        # data = [value for value in data if value != 0]
        # new_data = sparse.coo_matrix((data, (self.movielens['train'].row, self.movielens['train'].col)), shape=shape)
        pass
