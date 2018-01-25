import os
import urllib.request as req
from tqdm import tqdm

rootdir = 'val'

for folder in os.listdir(rootdir):
    curr_path = os.path.join(rootdir, folder)
    filename = 'urls'
    with open(os.path.join(curr_path, filename)) as urls:
        for idx, url in enumerate(tqdm(urls, desc=f'Current folder: {folder}')):
            filename = '_'.join([str(idx), url.split('/')[-1][:-1]])
            file = os.path.join(curr_path, filename)
            req.urlretrieve(url, file)
