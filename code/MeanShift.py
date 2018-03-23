import glob
from multiprocessing import Pool

import numpy as np
import tqdm
import scipy.ndimage
import os
from matplotlib import pyplot as plt
from PIL import Image

def meancalc(files):
    bo = Image.open(files[1])
    ao = Image.open(files[0])

    a = np.array(ao)
    b = np.array(bo)

    return (a.mean(),b.mean())

def meanapply(files):
    global p
    a = Image.open(files)
    a = np.array(a)
    a=p(a)
    Image.fromarray(np.uint8(a)).save('Corrected/DIFF/TRAIN_LR/'+os.path.basename(files))

def meanget(LR,HR):
    pool = Pool(8)
    xc = []
    yc = []
    for x,y in tqdm.tqdm(pool.imap_unordered(meancalc,zip(LR,HR)), total=len(HR)):
        xc.append(x)
        yc.append(y)
    return np.poly1d(np.polyfit(xc,yc,1))
