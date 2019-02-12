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
