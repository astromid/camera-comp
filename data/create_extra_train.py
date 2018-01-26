import os
import urllib.request as req
from tqdm import tqdm

TRAIN_DIR = 'train'
URLS_DIR = 'flickr_urls'

for folder in os.listdir(URLS_DIR):
    with open(os.path.join(URLS_DIR, folder, 'urls_final')) as urls_file, \
         open(os.path.join(URLS_DIR, folder, 'good_jpgs')) as good_jpgs:
        good_filenames = [filename[:-1] for filename in good_jpgs]
        urls = [url[:-1] for url in urls_file]
        count = 0
        for url in tqdm(urls, desc=f'Current folder: {folder}'):
            filename = url.split('/')[-1][:-1]
            if filename in good_filenames:
                count += 1
                filename_ = '_'.join([str(count), filename])
                file = os.path.join(TRAIN_DIR, folder, filename_)
                req.urlretrieve(url, file)
