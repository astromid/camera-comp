import os
import urllib.request as req
from tqdm import tqdm

VAL_DIR = 'val'
URLS_DIR = 'val_urls'

for folder in os.listdir(URLS_DIR):
    os.makedirs(os.path.join(VAL_DIR, folder))
    with open(os.path.join(URLS_DIR, folder, 'urls')) as urls:
        for idx, url in enumerate(tqdm(urls, desc=f'Current folder: {folder}')):
            filename = '_'.join([str(idx), url.split('/')[-1][:-1]])
            file = os.path.join(VAL_DIR, folder, filename)
            req.urlretrieve(url, file)


