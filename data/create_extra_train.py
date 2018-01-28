import os
import urllib.request as req
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

TRAIN_DIR = 'train'
URLS_DIR = 'flickr_urls'


def download(args_):
    idx, good_url, folder_ = args_
    filename = '_'.join([str(idx), good_url.split('/')[-1]])
    file = os.path.join(TRAIN_DIR, folder_, filename)
    if not os.path.exists(file):
        try:
            req.urlretrieve(good_url, file)
        except:
            os.remove(file)
            tqdm.write(f'Error while downloading file {file}')


for folder in os.listdir(URLS_DIR):
    with open(os.path.join(URLS_DIR, folder, 'urls')) as urls_file, \
            open(os.path.join(URLS_DIR, folder, 'good_jpgs')) as good_jpgs:
        good_filenames = sorted([filename[:-1] for filename in good_jpgs])
        urls = sorted([url[:-1] for url in urls_file])
        good_urls = []
        for url in urls:
            filename = url.split('/')[-1]
            if filename in good_filenames:
                good_urls.append(url)
                good_filenames.remove(filename)
        try:
            assert len(good_filenames) == 0
        except AssertionError:
            print(good_filenames)
            raise AssertionError
        total = len(good_urls)
        print(f'Found {total} good urls for {folder}')
        with tqdm(desc=f'Folder: {folder}', total=total) as pbar:
            with ThreadPool() as p:
                args = list(zip(np.arange(total), good_urls, [folder] * total))
                for _ in p.imap_unordered(download, args):
                    pbar.update()

