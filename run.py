from kingrec import KingRec
from kingrec.evaluation import precision_at_k
from kingrec.evaluation import recall_at_k
from kingrec.evaluation import auc_score
from kingrec.datasets import init_movielens
from lightfm.datasets import fetch_movielens

init_movielens('./dataset/ml-latest-small')

# movielens = fetch_movielens(genre_features=True)
movielens = init_movielens('./dataset/ml-latest-small')

train = movielens['train']
test = movielens['test']
item_features = movielens['item_features']

king_rec = KingRec(learning_rate=0.05, loss='warp')

model = king_rec.model
model.fit_partial(train, item_features=item_features, epochs=200, verbose=True, num_threads=16)

train_precision = precision_at_k(model, train, item_features=item_features, k=10).mean()
test_precision = precision_at_k(model, test, item_features=item_features, k=10).mean()

train_recall = recall_at_k(model, train, item_features=item_features, k=10).mean()
test_recall = recall_at_k(model, test, item_features=item_features, k=10).mean()

train_auc = auc_score(model, train, item_features=item_features).mean()
test_auc = auc_score(model, test, item_features=item_features).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('Recall: train %.2f, test %.2f.' % (train_recall, test_recall))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
