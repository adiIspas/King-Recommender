import logging
import os
from urllib import request

import requests
from bs4 import BeautifulSoup

tmdb_url_template = 'https://www.themoviedb.org/movie/${ID}/images/posters?image_language=en'
tmdb_url_template_no_language = 'https://www.themoviedb.org/movie/${ID}/images/posters?image_language=xx'

logging.getLogger().setLevel(logging.INFO)


def get_tmdb_posters(dataset_path, ids):
    total_to_process = len(ids)
    current_processed = 1
    exceptions = 0

    for internal_movie_id, url_movie_id in ids.items():
        movie_path = dataset_path + str(internal_movie_id) + "/"

        if os.path.exists(movie_path) and len(os.listdir(movie_path)) > 0:
            current_processed = current_processed + 1
            continue

        os.makedirs(movie_path, exist_ok=True)

        current_url = tmdb_url_template.replace("${ID}", str(url_movie_id))
        try:
            page = request.urlopen(current_url)
        except:
            exceptions = exceptions + 1
            continue

        soup = BeautifulSoup(page, 'html.parser')

        posters_ulr = soup.findAll('img', {"class": "poster"})
        if len(posters_ulr) == 1:
            current_url = tmdb_url_template_no_language.replace("${ID}", str(url_movie_id))
            page = request.urlopen(current_url)
            soup = BeautifulSoup(page, 'html.parser')

            posters_ulr = soup.findAll('img', {"class": "poster"})

        index = 1
        images_data = []
        for url in posters_ulr:
            logging.info('Process movie with id %s and poster number %s', internal_movie_id, index)

            if url.get('data-src'):
                try:
                    images_data.append(requests.get(url['data-src']).content)
                except:
                    exceptions = exceptions + 1
                    continue

            index = index + 1

        for img_data in images_data:
            with open(movie_path + str(index) + '.jpg', 'wb') as handler:
                handler.write(img_data)

        logging.info('Current progress: %s%% processed', round((current_processed / total_to_process) * 100, 2))
        current_processed = current_processed + 1

    logging.info('Number of exceptions %s', exceptions)
