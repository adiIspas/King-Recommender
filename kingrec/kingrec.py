from lightfm import LightFM

__all__ = ['KingRec']


class KingRec(object):
    def __init__(self, no_components=10, loss='warp', learning_rate=0.05):
        self.no_components = no_components
        self.loss = loss
        self.learning_rate = learning_rate

        self.model = LightFM(learning_rate=learning_rate, loss=loss)

    def fit(self, interactions, user_features=None, item_features=None,
            sample_weight=None, epochs=1, num_threads=1, verbose=False):

        self.model.fit(interactions, user_features=user_features, item_features=item_features,
                       sample_weight=sample_weight, epochs=epochs, num_threads=num_threads, verbose=verbose)
