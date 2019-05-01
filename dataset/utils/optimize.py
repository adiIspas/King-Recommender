import numpy as np
import time

from skopt import forest_minimize
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from kingrec.dataset import init_movielens


dataset = '../../../king-rec-dataset/ml-latest-small'
threads = 16
k = 10


def objective(params):
    epochs, learning_rate, no_components, item_alpha, scale = params

    user_alpha = item_alpha * scale

    model = LightFM(loss=loss, random_state=2019, learning_rate=learning_rate,
                    no_components=no_components, user_alpha=user_alpha, item_alpha=item_alpha)
    model.fit(train, item_features=item_features, epochs=epochs, num_threads=threads, verbose=True)

    patks = function_to_optimize(model, test, item_features=item_features, num_threads=threads)
    mapatk = np.mean(patks)

    # Make negative because we want to _minimize_ objective
    out = -mapatk

    # Handle some weird numerical shit going on
    if np.abs(out + 1) < 0.01 or out < -1.0:
        return 0.0
    else:
        return out


results = open('../../optimized-params-' + str(time.time()), 'a')
for features in [None, ['genres'], ['clusters'], ['genres', 'clusters']]:
    movielens = init_movielens(dataset, min_rating=3.5, k=k, item_features=features)
    train = movielens['train']
    test = movielens['test']
    item_features = movielens['item_features']

    for loss in ['bpr', 'warp']:
        for function_to_optimize in [precision_at_k, auc_score]:
            space = [(1, 2),  # epochs
                     (10 ** -4, 1.0, 'log-uniform'),  # learning_rate
                     (20, 200),  # no_components
                     (10 ** -6, 10 ** -1, 'log-uniform'),  # item_alpha
                     (0.001, 1., 'log-uniform'),  # user_scaling
                     ]

            current_fixed_params = '\n== Optimize with features: ' + str(features) + '   loss: ' + loss + \
                                   '   function to optimize: ' + function_to_optimize.__name__ + ' ==='
            print(current_fixed_params)
            results.write('\n' + current_fixed_params)

            res_fm = forest_minimize(objective, space, n_calls=10, random_state=7, verbose=True, n_jobs=-1)

            maximum_found = str('Maximum ' + function_to_optimize.__name__ + ' found: {:6.4f}'.format(-res_fm.fun))
            optimal_params = 'Optimal parameters:'

            print(maximum_found)
            print(optimal_params)

            results.write('\n' + maximum_found)
            results.write('\n' + optimal_params)

            params = ['epochs', 'learning_rate', 'no_components', 'item_alpha', 'scaling']
            for (p, x_) in zip(params, res_fm.x):
                params_values = str('{}: {}'.format(p, x_))

                print(params_values)
                results.write('\n' + params_values)

results.close()
