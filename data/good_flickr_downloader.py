import os
import urllib.request as req

rootdir = 'flickr_images'
outdir = 'good_flickr_out'
urls_fname = 'urls_final'
good_fname = os.path.join(rootdir, 'good_jpgs')

good_urls = []
with open(good_fname) as goods:
    for url in goods:
        good_urls.append(url.split('/')[-1])

for folder in os.listdir(rootdir):
    if os.path.isdir(os.path.join(rootdir, folder)):
        print (folder)
        with open(os.path.join(rootdir, folder, urls_fname)) as urls:
            if not os.path.exists(os.path.join(outdir, folder)):
                os.makedirs(os.path.join(outdir, folder))
            for idx, url in enumerate(urls):
                name = url.split('/')[-1]
                if name in good_urls:
                    filename = os.path.join(outdir, folder, name)
                    if not os.path.exists(filename):
                        req.urlretrieve(url, filename)
                #if idx > 100: 
                    #break