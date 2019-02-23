import csv


def save_internal_item_id_to_page_id(ids, path):
    with open(path, 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['item_internal_id', 'tmdb_id'])
        for key, value in ids.items():
            writer.writerow([str(key), str(value)])
