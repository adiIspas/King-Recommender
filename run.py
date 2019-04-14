from kingrec import KingRec
from kingrec.evaluation import precision_at_k
from kingrec.evaluation import recall_at_k
from kingrec.evaluation import auc_score
from kingrec.datasets import init_movielens
from lightfm.datasets import fetch_movielens

dataset = '../king-rec-dataset/ml-latest-small'

k_fold = 7

results = dict()
results['train_precision'] = 0
results['test_precision'] = 0
results['train_recall'] = 0
results['test_recall'] = 0
results['train_auc'] = 0
results['test_auc'] = 0

for idx in range(k_fold):
    # movielens = fetch_movielens(genre_features=True)
    movielens = init_movielens(dataset)

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

    results['train_precision'] = results['train_precision'] + train_precision
    results['test_precision'] = results['test_precision'] + test_precision
    results['train_recall'] = results['train_recall'] + train_recall
    results['test_recall'] = results['test_recall'] + test_recall
    results['train_auc'] = results['train_auc'] + train_auc
    results['test_auc'] = results['test_auc'] + test_auc

    print('\nResults for split number:', idx + 1)
    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('Recall: train %.2f, test %.2f.' % (train_recall, test_recall))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
    print('--------------------------------\n')


print('Final results')
print('Precision: train %.2f, test %.2f.' % (results['train_precision']/k_fold, results['test_precision']/k_fold))
print('Recall: train %.2f, test %.2f.' % (results['train_recall']/k_fold, results['test_recall']/k_fold))
print('AUC: train %.2f, test %.2f.' % (results['train_auc']/k_fold, results['test_auc']/k_fold))
print('--------------------------------')