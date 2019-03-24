from lightfm import LightFM

__all__ = ['KingRec']


class KingRec(object):
    def __init__(self, no_components=10, loss='warp', learning_rate=0.05):
        self.no_components = no_components
        self.loss = loss
        self.learning_rate = learning_rate

        self.model = LightFM(no_components=no_components, learning_rate=learning_rate, loss=loss, random_state=7)
