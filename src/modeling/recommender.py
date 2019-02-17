import logging

import numpy
import tensorrec
from scipy import sparse


class KingRecommender(object):

    def __init__(self, n_components=10,
                 user_repr_graph=tensorrec.representation_graphs.NormalizedLinearRepresentationGraph(),
                 item_repr_graph=tensorrec.representation_graphs.NormalizedLinearRepresentationGraph(),
                 prediction_graph=tensorrec.prediction_graphs.EuclideanSimilarityPredictionGraph(),
                 loss_graph=tensorrec.loss_graphs.RMSEDenseLossGraph()):
        self.model = tensorrec.TensorRec(n_components=n_components, user_repr_graph=user_repr_graph,
                                         item_repr_graph=item_repr_graph, prediction_graph=prediction_graph,
                                         loss_graph=loss_graph)

    def fit(self, user_features, item_features, interactions, epochs=300):
        logging.info('Fitting the model ...')

        self.model.fit(interactions=interactions, user_features=user_features, item_features=item_features,
                       n_sampled_items=int(item_features.shape[0] * .01), epochs=epochs, verbose=True)

    def recall_at_k(self, user_features, item_features, interactions, k=10):
        logging.info('Calculate recall at %s ...', k)

        ranks = self.model.predict_rank(user_features=user_features, item_features=item_features)
        recall_at_k = tensorrec.eval.recall_at_k(test_interactions=interactions,
                                                 predicted_ranks=ranks, k=k).mean()

        print("=== Recall at {0}: {1}% ===".format(k, round(recall_at_k * 100, 2)))

    def precision_at_k(self, user_features, item_features, interactions, k=10):
        logging.info('Calculate precision at %s ...', k)

        ranks = self.model.predict_rank(user_features=user_features, item_features=item_features)
        recall_at_k = tensorrec.eval.precision_at_k(test_interactions=interactions,
                                                    predicted_ranks=ranks, k=k).mean()

        print("=== Precision at {0}: {1}% ===".format(k, round(recall_at_k * 100, 2)))

    def recommend_for_user(self, user_id, titles, user_features, item_features, k=10):
        u_features = sparse.csr_matrix(user_features)[user_id]

        u_rankings = self.model.predict_rank(user_features=u_features,
                                             item_features=item_features)[0]

        u_top_k_recs = numpy.where(u_rankings <= k)[0]
        print("User {0} recommendations:".format(user_id))
        for m in u_top_k_recs:
            print("{0}".format(titles[m]))
