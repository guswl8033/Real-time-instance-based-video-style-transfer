import glob
import numpy as np
from PIL import Image

x = 0
r = 256
images = glob.glob('/mnt/ITRC/MPISintelDataset/flownet/clean/alley_1/*.png')
for i in images:
    img = Image.open(i)
    pix = np.array(img)
    h, w, c = pix.shape
    new_img = pix[(h -r) // 2:(h + r) // 2, (w - r) // 2:(w + r) // 2, :]
    final = Image.fromarray(new_img)
    final.save('/mnt/ITRC/MPISintelDataset/flownet/clear/'+ str(x) +'.png')
    x += 1