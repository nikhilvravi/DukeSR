import glob
from multiprocessing import Pool

import numpy as np
import tqdm
import scipy.ndimage
import os
from matplotlib import pyplot as plt
from PIL import Image
import MeanShift

LR = sorted(glob.glob("/home/superres98/SuperResolution/Corrected/TRAIN_LR_NOISE_MS/X4/*"))
HR = sorted(glob.glob("/home/superres98/SuperResolution/Corrected/TRAIN_LR_NOISE_MS/X4/*"))
TEST = sorted(glob.glob("/home/superres98/SuperResolution/Corrected/TRAIN_LR_NOISE_MS/X4/*"))

def bestshift(files):
    global p
    bo = Image.open(files)
    b = np.uint8(np.clip(p(np.array(b)),0,255))
    b = np.array(b)/255

    ima = Image.fromarray(b)
    ima.save("Corrected/DIFF/TRAIN_LR/"+os.path.basename(files))

print("Getting mean shift")
p=MeanShift.meanget(LR,HR)
print("Mean shift: "+str(p))

pool = Pool(8)
list(tqdm.tqdm(pool.imap_unordered(bestshift,TEST), total=len(HR)))
