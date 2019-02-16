import logging
import os
from urllib import request

import requests
from bs4 import BeautifulSoup

tmdb_url_template = 'https://www.themoviedb.org/movie/${ID}/images/posters?image_language=en'
logging.getLogger().setLevel(logging.INFO)


def get_tmdb_posters(dataset_path, ids):
    total_to_process = len(ids)
    current_processed = 1

    for internal_movie_id, url_movie_id in ids.items():
        movie_path = dataset_path + str(internal_movie_id) + "/"
        os.makedirs(movie_path)

        current_url = tmdb_url_template.replace("${ID}", str(url_movie_id))
        page = request.urlopen(current_url)
        soup = BeautifulSoup(page, 'html.parser')

        posters_ulr = soup.findAll('img', {"class": "poster"})

        index = 1
        for url in posters_ulr:
            logging.info('Process movie with id %s and poster number %s', internal_movie_id, index)

            if url.get('data-src'):
                img_data = requests.get(url['data-src']).content

                with open(movie_path + str(index) + '.jpg', 'wb') as handler:
                    handler.write(img_data)
                index = index + 1

        logging.info('Current progress: %s%% processed', round((current_processed / total_to_process) * 100, 2))
        current_processed = current_processed + 1
