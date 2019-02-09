import csv
import logging
from collections import defaultdict


def read_ratings(file_path):
    logging.info("Loading ratings ...")

    with open(file_path, 'r') as ratings_file:
        ratings_file_reader = csv.reader(ratings_file)
        raw_ratings = list(ratings_file_reader)
        raw_ratings.pop(0)  # remove header

    movielens_to_internal_user_ids = defaultdict(lambda: len(movielens_to_internal_user_ids))
    movielens_to_internal_item_ids = defaultdict(lambda: len(movielens_to_internal_item_ids))

    for row in raw_ratings:
        row[0] = movielens_to_internal_user_ids[int(row[0])]
        row[1] = movielens_to_internal_item_ids[int(row[1])]
        row[2] = float(row[2])

    n_users = len(movielens_to_internal_user_ids)
    n_items = len(movielens_to_internal_item_ids)

    return raw_ratings, n_users, n_items
