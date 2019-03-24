from kingrec import KingRec
from kingrec.evaluation import precision_at_k
from kingrec.evaluation import recall_at_k
from kingrec.evaluation import auc_score
from lightfm.datasets import fetch_movielens

movielens = fetch_movielens()
train = movielens['train']
test = movielens['test']

king_rec = KingRec(learning_rate=0.05, loss='warp')
king_rec.fit(train, epochs=200, verbose=True, num_threads=16)

train_precision = precision_at_k(king_rec.model, train, k=10).mean()
test_precision = precision_at_k(king_rec.model, test, k=10).mean()

train_recall = recall_at_k(king_rec.model, train, k=10).mean()
test_recall = recall_at_k(king_rec.model, test, k=10).mean()

train_auc = auc_score(king_rec.model, train).mean()
test_auc = auc_score(king_rec.model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('Recall: train %.2f, test %.2f.' % (train_recall, test_recall))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
