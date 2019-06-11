import csv
import os
import shutil

dataset = '../../../king-rec-dataset/ml-latest-small/'


def collect_first_poster():
    items_ids = list(get_items_ids())

    for item in items_ids:
        item = str(item)
        src = dataset + 'images/' + item + '/posters/' + '/1.jpg'
        dest = './images/' + item

        os.makedirs(dest, exist_ok=True)
        if os.path.isfile(src):
            shutil.copy(src, dest + '/1.jpg')


def get_items_ids():
    item_ids = set()

    with open(dataset + 'movies.csv', 'r') as movies_file:
        reader = csv.reader(movies_file, delimiter=',')
        next(reader)  # skip header

        for row in reader:
            item_ids.add(int(row[0]))

    return item_ids


collect_first_poster()
