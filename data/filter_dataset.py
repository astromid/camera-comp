import os
from glob import glob
from wand.image import Image
from tqdm import tqdm

files = glob('train/*/*')
delete_counter = 0
for file in tqdm(files):
    with Image(filename=file) as img:
        if img.compression_quality < 93:
            delete_counter += 1
            os.remove(file)
print(f'Delete {delete_counter} train samples')