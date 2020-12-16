import cv2
import numpy as np
import glob
import natsort

img_array = []
for filename in natsort.natsorted(glob.glob('/mnt/ITRC/yolact/results/final/warped/*.png')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('/mnt/ITRC/yolact/results/final/warped/project.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()