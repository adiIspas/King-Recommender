from lightfm import evaluation

__all__ = ['precision_at_k',
           'recall_at_k',
           'auc_score',
           'reciprocal_rank']


def recall_at_k(model, test_interactions, train_interactions=None,
                k=10, user_features=None, item_features=None,
                preserve_rows=False, num_threads=1, check_intersections=True):
    return evaluation.recall_at_k(model, test_interactions=test_interactions, train_interactions=train_interactions,
                                  k=k,
                                  user_features=user_features, item_features=item_features, preserve_rows=preserve_rows,
                                  num_threads=num_threads, check_intersections=check_intersections)


def precision_at_k(model, test_interactions, train_interactions=None,
                   k=10, user_features=None, item_features=None,
                   preserve_rows=False, num_threads=1, check_intersections=True):
    return evaluation.precision_at_k(model, test_interactions=test_interactions, train_interactions=train_interactions,
                                     k=k,
                                     user_features=user_features, item_features=item_features,
                                     preserve_rows=preserve_rows,
                                     num_threads=num_threads, check_intersections=check_intersections)


def auc_score(model, test_interactions, train_interactions=None, user_features=None, item_features=None,
              preserve_rows=False, num_threads=1, check_intersections=True):
    return evaluation.auc_score(model, test_interactions=test_interactions, train_interactions=train_interactions,
                                user_features=user_features, item_features=item_features, preserve_rows=preserve_rows,
                                num_threads=num_threads, check_intersections=check_intersections)


def reciprocal_rank(model, test_interactions, train_interactions=None, user_features=None, item_features=None,
                    preserve_rows=False, num_threads=1, check_intersections=True):
    return evaluation.reciprocal_rank(model, test_interactions=test_interactions, train_interactions=train_interactions,
                                      user_features=user_features, item_features=item_features,
                                      preserve_rows=preserve_rows,
                                      num_threads=num_threads, check_intersections=check_intersections)
