import tensorrec

from scipy import sparse


class KingRecommender(object):

    def __init__(self, dataset_base_path):
        self.dataset_base_path = dataset_base_path
        self.dataset_file_path = {}

        self.internal_users_ids = None
        self.internal_items_ids = None

        self.n_users = 0
        self.n_items = 0

        self.user_features = None
        self.item_features = None
        self.interactions = None
        self.test_interactions = None

        self.dataset = {}

        self.model = tensorrec.TensorRec(n_components=5, loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
        self.ranks = None

    def fit(self, epochs=300):
        self.model.fit(interactions=self.interactions,
                       user_features=self.user_features,
                       item_features=self.item_features,
                       n_sampled_items=int(self.n_items * .01),
                       epochs=epochs,
                       verbose=True)

    def check_results(self):
        self.ranks = self.model.predict_rank(user_features=self.user_features, item_features=self.item_features)

        train_recall_at_10 = tensorrec.eval.recall_at_k(
            test_interactions=self.interactions,
            predicted_ranks=self.ranks,
            k=10).mean()
        test_recall_at_10 = tensorrec.eval.recall_at_k(
            test_interactions=self.test_interactions,
            predicted_ranks=self.ranks,
            k=10).mean()

        print("=== Recall at 10: Train: {:.4f} Test: {:.4f} ==="
              .format(train_recall_at_10 * 100, test_recall_at_10 * 100))

    def add_dataset_file_path(self, file_key, file_name):
        self.dataset_file_path[file_key] = self.dataset_base_path + file_name

    def get_dataset_file_path(self, file_key):
        return self.dataset_file_path[file_key]

    def add_internal_user_ids(self, internal_user_ids):
        self.internal_users_ids = internal_user_ids
        self.n_users = len(internal_user_ids)
        self.user_features = sparse.identity(self.n_users)

    def add_internal_item_ids(self, internal_item_ids):
        self.internal_items_ids = internal_item_ids
        self.n_items = len(internal_item_ids)
        self.item_features = sparse.identity(self.n_items)

    def add_dataset_entry(self, entry_key, entry):
        self.dataset[entry_key] = entry

    def get_dataset_entry(self, entry_key):
        return self.dataset[entry_key]

    def add_interactions(self, interactions):
        self.interactions = interactions

    def add_test_interactions(self, test_interactions):
        self.test_interactions = test_interactions

    def add_user_features(self, user_features):
        self.user_features = user_features

    def add_item_features(self, item_features):
        self.item_features = item_features
