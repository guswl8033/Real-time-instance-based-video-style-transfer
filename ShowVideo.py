import os
import matplotlib.image as img
import cv2
import numpy as np
import time
from PIL import Image
import os.path

originpath = '/mnt/ITRC/flownet2-pytorch/dataset/test/cropped/'
transferpath = '/mnt/ITRC/ITRC/roix/test/final/'
framenum = 0

while(True):
    if not os.path.isfile(transferpath+'%03d'%framenum+'_person0_roi.npy') :
        break
    start1 = time.time()
    ori= img.imread(originpath+'frame%04d'%framenum+'.jpg')
    transfer=np.load(transferpath+'%03d'%framenum+'_person0_roi.npy')
    start2 = time.time()
    print('imread:', start2-start1)
    ori=cv2.resize(ori, dsize=(750, 400), interpolation=cv2.INTER_CUBIC)
    transfer=cv2.resize(transfer, dsize=(750, 400), interpolation=cv2.INTER_CUBIC)
    start3 = time.time()
    print('resize:', start3-start2)
    ori = ori[..., ::-1]
    cv2.imshow('origin',ori)
    cv2.moveWindow('origin',0,0)
    cv2.imshow('result', transfer)
    cv2.moveWindow('result', 0,500)
    start4=time.time()
    print('imshow:', start4-start3)

    framenum += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, close windows
cv2.destroyAllWindows()

