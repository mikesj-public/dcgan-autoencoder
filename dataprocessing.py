#!/usr/bin/env python

from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
from time import time
import sys
import h5py
from tqdm import tqdm


image_dir = 'img_align_celeba/'
try:
    image_locs = [join(image_dir,f) for f in listdir(image_dir) if isfile(join(image_dir,f))]
except:
    print "expected aligned images directory, see README"

total_imgs = len(image_locs)
print "found %i images in directory" %total_imgs

def process_image(im):
    if im.mode != "RGB":
        im = im.convert("RGB")
    new_size = [int(i/1.3) for i in im.size]
    im.thumbnail(new_size, Image.ANTIALIAS)
    target = np.array(im)[3:-3,4:-4,:]
    im = Image.fromarray(target)
    new_size = [i/4 for i in im.size]
    im.thumbnail(new_size, Image.ANTIALIAS)
    input = np.array(im)
    return input, target


def proc_loc(loc):
    try:
        i = Image.open(loc)
        input, target = process_image(i)
        return (input, target)
    except KeyboardInterrupt:
        raise
    except:
        return None 


try:
    hf = h5py.File('faces.hdf5','r+')
except:
    hf = h5py.File('faces.hdf5','w')


try:
    dset_t = hf.create_dataset("target", (1,160,128,3), 
                               maxshape= (1e6,160,128,3), chunks = (1,160,128,3), compression = "gzip") 
except:
    dset_t = hf['target']

try:
    dset_i = hf.create_dataset("input", (1, 40, 32, 3), 
                               maxshape= (1e6, 40, 32, 3), chunks = (1, 40, 32, 3), compression = "gzip")
except:
    dset_i = hf['input']

batch_size = 1024
num_iter = total_imgs / 1024

insert_point = 0
print "STARTING PROCESSING IN BATCHES OF %i" %batch_size

for i in tqdm(range(num_iter)):
    sys.stdout.flush()

    X_in  = []
    X_ta = []

    a = time()
    locs = image_locs[i * batch_size : (i + 1) * batch_size]

    proc =  [proc_loc(loc) for loc in locs]

    for pair in proc:
        if pair is not None:
            input, target = pair
            X_in.append(input)
            X_ta.append(target)

    X_in = np.array(X_in)
    X_ta = np.array(X_ta)

    dset_i.resize((insert_point + len(X_in),40, 32, 3))
    dset_t.resize((insert_point + len(X_in),160,128,3))

    dset_i[insert_point:insert_point + len(X_in)] = X_in
    dset_t[insert_point:insert_point + len(X_in)] = X_ta

    insert_point += len(X_in)
    
hf.close()
