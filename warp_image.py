import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import os
import time
from PIL import Image
import natsort
import imageio

import argparse

#******************** Get options for training********************#

parser = argparse.ArgumentParser(description = 'Warping argument')

parser.add_argument('--img_reference', metavar = 'ref', type=str, default='ref', help = 'reference high resolution image')
parser.add_argument('--flo', metavar = 'flo', type=str, default='flo', help = 'flo reference to target')
parser.add_argument('--result_root',dest='result_root',  type=str, default='./results/test', help='root to save warp image')
parser.add_argument('--result_dir',dest='result_dir',  type=str, default='./results/test/test_result.png', help='directory to save warp image')

args = parser.parse_args()

def get_full_path(dataset_path):
    """
    Get full path of data based on configs and target path
    example: datasets/train    """

    return os.path.join(dataset_path)

def warp(x, flo,i):

    B, C, H, W = x.shape
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    if x.is_cuda:
        grid = grid.cuda()

    vgrid = Variable(grid) + flo

    # scale grid to [-1,1](almost)
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)

    hsv = np.zeros((1024, 1920, 3))

    output = nn.functional.grid_sample(x, vgrid, padding_mode='border')

    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    temp= output*mask

    return output*mask, mask



def main():
    source_dir='/mnt/ITRC/ITRC/roix/test/final'
    for i in range(0,343):
        start_time=time.time()
        #source_root = get_full_path(args.img_reference)
        if i%5==4:
            continue
        if i%5==0:
            source_root = source_dir+'/%03d_person0_roi.npy' % i

        # flo_root = get_full_path(args.flo)
        flo_root = '/mnt/ITRC/flownet2-pytorch/testC/inference/run.epoch-0-flow-field/%06d'%i + '.flo'
        flo = cv2.readOpticalFlow(flo_root)
        flo = flo*(-1)
        #
        # flo_root = '/mnt/ITRC/flownet2-pytorch/dataset/VKitti_OF/Scene01/15-deg-left/frames/forwardFlow/Camera_0/flow_%05d.png' % i
        # flo = read_vkitti_png_flow(flo_root, i)


        source = np.load(source_root)[:1024,...]
    #    flo=np.load(flo_root)

        source = torch.tensor(source)
        flo = torch.tensor(flo)

        source=source.permute(2,0,1)
        flo=flo.permute(2,0,1)

        source=source.unsqueeze(0)
        flo=flo.unsqueeze(0) #(1,3,h,w)

        warp_source, mask= warp(source.float().cuda(), flo.float().cuda(), i)

        warp_source = warp_source.squeeze(0)
        warp_source = warp_source.permute(1,2,0)
        warp_source = warp_source.cpu().detach().numpy()
        warp_source = warp_source.astype(np.uint8)


        result_dir = source_dir + '/%03d_person0_roi.npy' % (i+1)

        # result_dir = './final/'+str(i+1)+'.png'
        source_root = result_dir
        # cv2.imwrite(result_dir, warp_source)
        np.save(result_dir[:-4],warp_source)
        #pdb.set_trace()

        print("successfully warped-frame:%03d"%i)
        print(time.time()-start_time)

    # gif = [f"/mnt/ITRC/ITRC/roix/test/gif/{i}" for i in
    #            natsort.natsorted(os.listdir('/mnt/ITRC/ITRC/roix/test/gif/'))]
    # gifs = [Image.open(i) for i in gif]
    # imageio.mimsave('/mnt/ITRC/ITRC/roix/test/gif/test.gif', gifs, fps=30)


if __name__ == '__main__':

    main()


