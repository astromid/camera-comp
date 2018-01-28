import os
import urllib.request as req
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

VAL_DIR = 'val'
URLS_DIR = 'val_urls'


def download(args_):
    idx, good_url, folder_ = args_
    filename = '_'.join([str(idx), good_url.split('/')[-1]])
    file = os.path.join(VAL_DIR, folder_, filename)
    if not os.path.exists(file):
        try:
            req.urlretrieve(good_url, file)
        except:
            os.remove(file)
            tqdm.write(f'Error while downloading file {file}')


for folder in os.listdir(URLS_DIR):
    os.makedirs(os.path.join(VAL_DIR, folder))
    with open(os.path.join(URLS_DIR, folder, 'urls')) as urls_file:
        urls = sorted([url[:-1] for url in urls_file])
        total = len(urls)
        print(f'Found {total} urls for {folder}')
        with tqdm(desc=f'Folder: {folder}', total=total) as pbar:
            with ThreadPool() as p:
                args = list(zip(np.arange(total), urls, [folder] * total))
                for _ in p.imap_unordered(download, args):
                    pbar.update()


