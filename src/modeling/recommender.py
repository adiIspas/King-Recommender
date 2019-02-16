import logging

import numpy
import tensorrec
from scipy import sparse


class KingRecommender(object):

    def __init__(self, user_features, item_features, train_interactions, test_interactions):
        self.user_features = user_features
        self.item_features = item_features

        self.interactions = train_interactions
        self.test_interactions = test_interactions

        self.model = tensorrec.TensorRec(n_components=5, loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
        self.ranks = None

    def fit(self, epochs=300):
        logging.info('Fitting the model ...')
        self.model.fit(interactions=self.interactions,
                       user_features=self.user_features,
                       item_features=self.item_features,
                       n_sampled_items=int(self.item_features.shape[0] * .01),
                       epochs=epochs,
                       verbose=True)

    def recall_at_k(self, k=10):
        logging.info('Calculate recall at %s ...', k)
        self.ranks = self.model.predict_rank(user_features=self.user_features, item_features=self.item_features)

        train_recall_at_k = tensorrec.eval.recall_at_k(
            test_interactions=self.interactions,
            predicted_ranks=self.ranks,
            k=10).mean()
        test_recall_at_k = tensorrec.eval.recall_at_k(
            test_interactions=self.test_interactions,
            predicted_ranks=self.ranks,
            k=10).mean()

        print("=== Recall at {0}: Train: {1}%% Test: {2}%% ==="
              .format(k, round(train_recall_at_k * 100, 2), round(test_recall_at_k * 100, 2)))

    def recommend_for_user(self, user_id, movie_titles, k=10):
        u_features = sparse.csr_matrix(self.user_features)[user_id]

        u_rankings = self.model.predict_rank(user_features=u_features,
                                             item_features=self.item_features)[0]

        u_top_k_recs = numpy.where(u_rankings <= k)[0]
        print("User {0} recommendations:".format(user_id))
        for m in u_top_k_recs:
            print("{0}".format(movie_titles[m]))
