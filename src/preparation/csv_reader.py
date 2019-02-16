import csv
import logging
from collections import defaultdict


def create_internal_ids(file_path):
    logging.info('Creating internal ids ...')

    with open(file_path, 'r') as file:
        file_reader = csv.reader(file)
        raw_file = list(file_reader)
        raw_file.pop(0)  # remove header

    internal_user_ids = defaultdict(lambda: len(internal_user_ids))
    internal_item_ids = defaultdict(lambda: len(internal_item_ids))

    for row in raw_file:
        internal_user_ids[int(row[0])]
        internal_item_ids[int(row[1])]

    return internal_user_ids, internal_item_ids


def read_ratings(file_path, internal_user_ids, internal_item_ids):
    logging.info("Loading ratings ...")

    with open(file_path, 'r') as ratings_file:
        ratings_file_reader = csv.reader(ratings_file)
        raw_ratings = list(ratings_file_reader)
        raw_ratings.pop(0)  # remove header

    for row in raw_ratings:
        row[0] = internal_user_ids[int(row[0])]
        row[1] = internal_item_ids[int(row[1])]
        row[2] = float(row[2])

    return raw_ratings


def read_genres(file_path, internal_item_ids):
    logging.info("Loading generes ...")

    with open(file_path, 'r') as genres_file:
        genres_file_reader = csv.reader(genres_file)
        raw_genres = list(genres_file_reader)
        raw_genres.pop(0)  # remove header

    movie_genres_by_internal_id = {}
    movie_titles_by_internal_id = {}

    for row in raw_genres:
        row[0] = internal_item_ids[int(row[0])]
        row[2] = row[2].split('|')

        movie_titles_by_internal_id[row[0]] = row[1]
        movie_genres_by_internal_id[row[0]] = row[2]

    return movie_titles_by_internal_id, movie_genres_by_internal_id
