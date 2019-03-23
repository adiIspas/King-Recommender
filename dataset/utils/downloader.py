import os
import csv
import sys
import requests
import urllib.request

api_key = sys.argv[1]

dataset = '../ml-latest-small/'
tmdb_api = 'https://api.themoviedb.org/3/movie/$MOVIE_ID/images?api_key=$API_KEY'
tmdb_images_url = 'https://image.tmdb.org/t/p/original/'


def get_tmdb_posters(tmdb_api_key):
    tmdb_movies_id = get_tmdb_ids()
    download_images(tmdb_api_key, tmdb_movies_id)

    return tmdb_movies_id


def download_images(tmdb_api_key, tmdb_movies_id):
    images = dataset + 'images/'
    posters = images + 'posters/'
    backdrops = images + 'backdrops/'

    idx = 1
    for key, value in tmdb_movies_id.items():
        posters = images + str(key) + '/posters/'
        backdrops = images + str(key) + '/backdrops/'

        current_url = tmdb_api.replace('$MOVIE_ID', str(value)).replace('$API_KEY', tmdb_api_key)
        response = requests.get(current_url)

        if response.status_code == 200:
            json = response.json()

            if not os.path.exists(posters):
                os.makedirs(posters)

            image_idx = 1
            for poster in json['posters']:
                print('Process movie', value, 'and poster', image_idx)
                poster_url = poster['file_path']
                urllib.request.urlretrieve(tmdb_images_url + poster_url, posters + str(image_idx) + '.jpg')
                image_idx += 1

            if not os.path.exists(backdrops):
                os.makedirs(backdrops)

            image_idx = 1
            for backdrop in json['backdrops']:
                print('Process movie', value, 'and backdrop', image_idx)
                backdrop_url = backdrop['file_path']
                urllib.request.urlretrieve(tmdb_images_url + backdrop_url, backdrops + str(image_idx) + '.jpg')
                image_idx += 1

        else:
            print('Status code:', response.status_code, 'on movie', value)

        if idx == 10:
            break

        idx += 1


def get_tmdb_ids(tmdb_index=2):
    links = dataset + 'links.csv'
    with open(links, 'r') as links_file:
        reader = csv.reader(links_file, delimiter=',', )
        next(reader)  # skip header

        tmdb_movies_id = dict()
        for row in reader:
            tmdb_movies_id.update({row[0]: row[tmdb_index]})
            
    return tmdb_movies_id


get_tmdb_posters(api_key)
