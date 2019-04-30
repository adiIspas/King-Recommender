from kingrec import KingRec
from kingrec.evaluation import precision_at_k
from kingrec.evaluation import recall_at_k
from kingrec.evaluation import auc_score
from kingrec.datasets import init_movielens
import matplotlib.pyplot as plt
import numpy as np

dataset = '../king-rec-dataset/ml-latest-small'
epochs = 10
k = 3
movielens = init_movielens(dataset)

train = movielens['train']
test = movielens['test']
item_features = movielens['item_features']

king_rec = KingRec(no_components=25)
model = king_rec.model

train_auc_scores = []
test_auc_scores = []

train_recall_scores = []
train_precision_scores = []

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

    train_recall_scores.append(train_recall)
    train_precision_scores.append(train_precision)


x = np.arange(len(train_recall_scores))
plt.plot(x, np.array(train_recall_scores))
plt.plot(x, np.array(train_precision_scores))
plt.legend(['train recall', 'train precision'], loc='lower right')
plt.show()
