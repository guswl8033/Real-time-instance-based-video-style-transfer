import numpy as np
from os.path import *
import matplotlib.pyplot as pp
import matplotlib.image as img
from PIL import Image

#from scipy.misc import imread
import imageio

im = img.imread('/mnt/Wiset_hyeonji/Dataset/Shanghaitech/testing/RGB/02_0128/198.jpg')
im = np.array(im)
pp.imshow(im)
pp.show()
npad = ((100, 100), (100, 100), (0,0))
im = np.pad(im, npad, 'constant', constant_values=0)
im = Image.fromarray(im)
pp.imshow(im)
pp.show()