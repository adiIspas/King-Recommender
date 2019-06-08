import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc
from kingrec import KingRec
from kingrec.evaluation import precision_at_k
from kingrec.evaluation import recall_at_k
from kingrec.evaluation import auc_score
from kingrec.evaluation import reciprocal_rank
from kingrec.dataset import init_movielens


def load_params(optimized_for='auc_clusters'):
    if optimized_for == 'auc_clusters':
        optimal_epochs = 100
        optimal_learning_rate = 0.0570326091236193
        optimal_no_components = 68
        optimal_alpha = 0.0029503539747277366
        optimal_scaling = 0.02563602355611453

        return optimal_epochs, optimal_learning_rate, optimal_no_components, optimal_alpha, optimal_scaling


k = 5
threads = 16
dataset = '../king-rec-dataset/ml-latest-small'
epochs, learning_rate, no_components, alpha, scaling = load_params(optimized_for='auc_clusters')

movielens = init_movielens(dataset, min_rating=3.5, k=k, item_features=['clusters'])

train = movielens['train']
test = movielens['test']
item_features = movielens['item_features']

king_rec = KingRec(no_components=no_components, learning_rate=learning_rate, alpha=alpha, scale=scaling, loss='warp')

model = king_rec.model
model.fit_partial(interactions=train, item_features=item_features, epochs=epochs, verbose=True, num_threads=threads)

train_precision = precision_at_k(model, train, item_features=item_features, k=k, num_threads=threads).mean()
test_precision = precision_at_k(model, test, item_features=item_features, k=k, num_threads=threads).mean()

train_recall = recall_at_k(model, train, item_features=item_features, k=k, num_threads=threads).mean()
test_recall = recall_at_k(model, test, item_features=item_features, k=k, num_threads=threads).mean()

train_auc = auc_score(model, train, item_features=item_features, num_threads=threads).mean()
test_auc = auc_score(model, test, item_features=item_features, num_threads=threads).mean()

train_reciprocal = reciprocal_rank(model, train, item_features=item_features, num_threads=threads).mean()
test_reciprocal = reciprocal_rank(model, test, item_features=item_features, num_threads=threads).mean()

print('\nResults')
print('AUC: train %.4f, test %.4f.' % (train_auc, test_auc))
print('Recall: train %.4f, test %.4f.' % (train_recall, test_recall))
print('Precision: train %.4f, test %.4f.' % (train_precision, test_precision))
print('Reciprocal rank: train %.4f, test %.4f.' % (train_reciprocal, test_reciprocal))
print('--------------------------------\n')

# plot precision/recall graph
test_recall = recall_at_k(model, test, item_features=item_features, k=k, num_threads=threads)
test_precision = precision_at_k(model, test, item_features=item_features, k=k, num_threads=threads)

sorted_idx = test_recall.argsort()
test_recall = test_recall[sorted_idx]

test_precision = test_precision[sorted_idx]
auc = np.round(auc(test_recall, test_precision), decimals=2)

plt.title('AUC: ' + str(auc) + '%')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(test_recall, test_precision)
plt.savefig('precision-recall.jpg')
plt.show()
