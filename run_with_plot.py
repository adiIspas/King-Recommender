from kingrec import KingRec
from kingrec.evaluation import precision_at_k
from kingrec.evaluation import recall_at_k
from kingrec.evaluation import auc_score
from kingrec.dataset import init_movielens
import matplotlib.pyplot as plt
import numpy as np

dataset = '../king-rec-dataset/ml-latest-small'

epochs = 100
learning_rate = 0.00992574866043483
no_components = 196
alpha = 1.4998416303979942e-05
scaling = 0.0012546879899490554
k = 3

movielens = init_movielens(dataset, min_rating=3.5, k=k)

train = movielens['train']
test = movielens['test']
item_features = movielens['item_features']

king_rec = KingRec(no_components=no_components, learning_rate=learning_rate, alpha=alpha, loss='warp')
model = king_rec.model

train_auc_scores = []
test_auc_scores = []

train_precision_scores = []
test_precision_scores = []

train_recall_scores = []
test_recall_scores = []

for epoch in range(epochs):
    print('Epoch:', epoch)
    model.fit_partial(train, item_features=item_features, epochs=1)

    train_precision = precision_at_k(model, train, item_features=item_features, k=k).mean()
    test_precision = precision_at_k(model, test, item_features=item_features, k=k).mean()

    train_recall = recall_at_k(model, train, item_features=item_features, k=k).mean()
    test_recall = recall_at_k(model, test, item_features=item_features, k=k).mean()

    train_auc = auc_score(model, train, item_features=item_features).mean()
    test_auc = auc_score(model, test, item_features=item_features).mean()

    train_auc_scores.append(train_auc)
    test_auc_scores.append(test_auc)

    train_precision_scores.append(train_precision)
    test_precision_scores.append(test_precision)

    train_recall_scores.append(train_recall)
    test_recall_scores.append(test_recall)

x = np.arange(len(train_auc_scores))
plt.plot(x, np.array(train_auc_scores))
plt.plot(x, np.array(test_auc_scores))
plt.legend(['train acc', 'test acc'], loc='lower right')
plt.savefig('acc.png')
plt.show()

x = np.arange(len(train_precision_scores))
plt.plot(x, np.array(train_precision_scores))
plt.plot(x, np.array(test_precision_scores))
plt.legend(['train precision', 'test precision'], loc='lower right')
plt.savefig('precision.png')
plt.show()

x = np.arange(len(train_recall_scores))
plt.plot(x, np.array(train_recall_scores))
plt.plot(x, np.array(test_recall_scores))
plt.legend(['train recall', 'test recall'], loc='lower right')
plt.savefig('recall.png')
plt.show()
