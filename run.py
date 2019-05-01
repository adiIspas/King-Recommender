from kingrec import KingRec
from kingrec.evaluation import precision_at_k
from kingrec.evaluation import recall_at_k
from kingrec.evaluation import auc_score
from kingrec.evaluation import reciprocal_rank
from kingrec.dataset import init_movielens

dataset = '../king-rec-dataset/ml-latest-small'

# without features
# epochs = 254
# learning_rate = 0.003757852437153106
# no_components = 114
# alpha = 0.06690348539114543
# scaling = 0.001504707940775378

# genres
# epochs = 158
# learning_rate = 0.015453475642479833
# no_components = 130
# alpha = 0.0007855993801387813
# scaling = 0.8403398223850091

# clusters
epochs = 55
learning_rate = 0.00992574866043483
no_components = 196
alpha = 1.4998416303979942e-05
scaling = 0.0012546879899490554

# clusters + genres
# epochs = 183
# learning_rate = 0.037082502295401325
# no_components = 21
# alpha = 0.0014490168877726135
# scaling = 0.07389666871280426

k = 3
threads = 16

movielens = init_movielens(dataset, min_rating=3.5, k=k, item_features=['clusters'])

train = movielens['train']
test = movielens['test']
item_features = movielens['item_features']

king_rec = KingRec(no_components=no_components, learning_rate=learning_rate, alpha=alpha, scale=scaling, loss='warp')

model = king_rec.model
model.fit_partial(train, item_features=item_features, epochs=epochs, verbose=True, num_threads=threads)

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
