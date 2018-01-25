import os
import urllib.request as req

rootdir = 'flickr_images'
outdir = 'flickr_out'
urls_fname = "urls_final"

for folder in os.listdir(rootdir):
    if os.path.isdir(os.path.join(rootdir, folder)):
        print (folder)
        with open(os.path.join(rootdir, folder, urls_fname)) as urls:
            if not os.path.exists(os.path.join(outdir, folder)):
                os.makedirs(os.path.join(outdir, folder))
            for idx, url in enumerate(urls):
                filename = os.path.join(outdir, folder, url.split('/')[-1])
                if not os.path.exists(filename):
                    req.urlretrieve(url, filename)
                #if idx > 100: 
                    #break