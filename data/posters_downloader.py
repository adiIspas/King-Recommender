import logging
import os
from urllib import request

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

tmdb_url_template = 'https://www.themoviedb.org/movie/${ID}/images/posters?image_language=en'
tmdb_url_template_no_language = 'https://www.themoviedb.org/movie/${ID}/images/posters?image_language=xx'

logging.getLogger().setLevel(logging.INFO)


def retry_session(retries, session=None, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504)):
    session = session or requests.Session()
    retry = Retry(total=retries,
                  read=retries,
                  connect=retries,
                  backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_tmdb_posters(dataset_path, ids):
    total_to_process = len(ids)
    current_processed = 1
    exceptions = 0
    session = retry_session(retries=10)

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
            logging.error('Error for url %s and internal id %s', current_url, internal_movie_id)
            exceptions = exceptions + 1
            continue

        soup = BeautifulSoup(page, 'html.parser')

        posters_ulr = soup.findAll('img', {"class": "poster"})
        if len(posters_ulr) == 1:
            current_url = tmdb_url_template_no_language.replace("${ID}", str(url_movie_id))
            try:
                page = request.urlopen(current_url)
            except:
                logging.error('Error for url %s and internal id %s', current_url, internal_movie_id)
                exceptions = exceptions + 1
                continue
            soup = BeautifulSoup(page, 'html.parser')

            posters_ulr = soup.findAll('img', {"class": "poster"})

        index = 0
        images_data = []
        for url in posters_ulr:

            if url.get('data-src'):
                images_data.append(session.get(url['data-src']).content)
                index = index + 1

            logging.info('Process movie with id %s and poster number %s', internal_movie_id, index)

        index = 1
        for img_data in images_data:
            with open(movie_path + str(index) + '.jpg', 'wb') as handler:
                handler.write(img_data)
            index = index + 1

        logging.info('Current progress: %s%% processed', round((current_processed / total_to_process) * 100, 2))
        current_processed = current_processed + 1

    logging.info('Number of exceptions %s', exceptions)
