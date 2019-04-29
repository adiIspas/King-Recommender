from skopt import forest_minimize
import numpy as np
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from kingrec.datasets import init_movielens

dataset = '../king-rec-dataset/ml-latest-small'
movielens = init_movielens(dataset, min_rating=0.0)
train = movielens['train']
test = movielens['test']


def objective(params):
    # unpack
    epochs, learning_rate, no_components, alpha = params

    user_alpha = alpha
    item_alpha = alpha
    model = LightFM(loss='warp',
                    random_state=2019,
                    learning_rate=learning_rate,
                    no_components=no_components,
                    user_alpha=user_alpha,
                    item_alpha=item_alpha)
    model.fit(train, epochs=epochs, num_threads=16, verbose=True)

    patks = precision_at_k(model, test, train_interactions=None, k=5, num_threads=4)
    mapatk = np.mean(patks)
    # Make negative because we want to _minimize_ objective
    out = -mapatk
    # Handle some weird numerical shit going on
    if np.abs(out + 1) < 0.01 or out < -1.0:
        return 0.0
    else:
        return out


space = [(1, 260),  # epochs
         (10 ** -4, 1.0, 'log-uniform'),  # learning_rate
         (20, 200),  # no_components
         (10 ** -6, 10 ** -1, 'log-uniform'),  # alpha
         ]

res_fm = forest_minimize(objective, space, n_calls=250, random_state=0, verbose=True)

print('Maximum p@k found: {:6.5f}'.format(-res_fm.fun))
print('Optimal parameters:')
params = ['epochs', 'learning_rate', 'no_components', 'alpha']
for (p, x_) in zip(params, res_fm.x):
    print('{}: {}'.format(p, x_))
