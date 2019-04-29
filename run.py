from kingrec import KingRec
from kingrec.evaluation import precision_at_k
from kingrec.evaluation import recall_at_k
from kingrec.evaluation import auc_score
from kingrec.datasets import init_movielens

dataset = '../king-rec-dataset/ml-latest-small'

learning_rate = 0.05367215567275274
no_components = 184
alpha = 0.002064592490166775
epochs = 243
k = 5
threads = 16

movielens = init_movielens(dataset, min_rating=2.5)

train = movielens['train']
test = movielens['test']
item_features = movielens['item_features']
user_features = movielens['user_features']
# item_features = None
# user_features = None

king_rec = KingRec(no_components=no_components, learning_rate=learning_rate, alpha=alpha, loss='warp')

model = king_rec.model
model.fit_partial(train, user_features=user_features, item_features=item_features, epochs=epochs, verbose=True, num_threads=16)

train_precision = precision_at_k(model, train, user_features=user_features, item_features=item_features, k=k).mean()
test_precision = precision_at_k(model, test, user_features=user_features, item_features=item_features, k=k).mean()

train_recall = recall_at_k(model, train, user_features=user_features, item_features=item_features, k=k).mean()
test_recall = recall_at_k(model, test, user_features=user_features, item_features=item_features, k=k).mean()

train_auc = auc_score(model, train, user_features=user_features, item_features=item_features).mean()
test_auc = auc_score(model, test, user_features=user_features, item_features=item_features).mean()

print('\nResults')
print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('Recall: train %.2f, test %.2f.' % (train_recall, test_recall))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
print('--------------------------------\n')
