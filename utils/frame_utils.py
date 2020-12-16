import numpy as np
from os.path import *
import matplotlib.pyplot as pp
import matplotlib.image as img
from PIL import Image

#from scipy.misc import imread
from . import flow_utils 
import imageio
def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = img.imread(file_name)
        # im = Image.open(file_name)
        # im = np.array(im)
        # im = im[0:1024, : , :]
        # im = Image.fromarray(im)
        # #
        # im.save(file_name[:-14]+'/cropped/frame'+file_name[-8:])
        # im = np.array(im)
        #
        # npad = ((100, 100), (100, 100),(0,0))
        # im = np.pad(im, npad, 'constant', constant_values=0)

        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    return []
