import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.utils import makeVideo
import torch.backends.cudnn as cudnn
from libs.models import encoder3, encoder4
from libs.models import decoder3, decoder4
import torchvision.transforms as transforms
import torch.nn.functional as nnf
from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import copy
from skimage.transform import resize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse, os, sys, subprocess
import setproctitle, colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

import models, losses, datasets
from utils import flow_utils, tools

# fp32 copy of parameters for update
global param_copy


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(
    description='YOLACT COCO Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to evaulate model')
parser.add_argument('--fast_nms', default=True, type=str2bool,
                    help='Whether to use a faster, but not entirely correct version of NMS.')
parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                    help='Whether compute NMS cross-class or per-class.')
parser.add_argument('--display_masks', default=True, type=str2bool,
                    help='Whether or not to display masks over bounding boxes')
parser.add_argument('--display_bboxes', default=True, type=str2bool,
                    help='Whether or not to display bboxes around masks')
parser.add_argument('--display_text', default=True, type=str2bool,
                    help='Whether or not to display text (class [score])')
parser.add_argument('--display_scores', default=True, type=str2bool,
                    help='Whether or not to display scores in addition to classes')
parser.add_argument('--display', dest='display', action='store_true',
                    help='Display qualitative results instead of quantitative ones.')
parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                    help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                    help='In quantitative mode, the file to save detections before calculating mAP.')
parser.add_argument('--resume_y', dest='resume_y', action='store_true',
                    help='If display not set, this resumes mAP calculations from the ap_data_file.')
parser.add_argument('--max_images', default=-1, type=int,
                    help='The maximum number of images from the dataset to consider. Use -1 for all.')
parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                    help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                    help='The output file for coco bbox results if --coco_results is set.')
parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                    help='The output file for coco mask results if --coco_results is set.')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                    help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
parser.add_argument('--web_det_path', default='web/dets/', type=str,
                    help='If output_web_json is set, this is the path to dump detections into.')
parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                    help='Do not output the status bar. This is useful for when piping to a file.')
parser.add_argument('--display_lincomb', default=False, type=str2bool,
                    help='If the config uses lincomb masks, output a visualization of how those masks are created.')
parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                    help='Equivalent to running display mode but without displaying an image.')
parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                    help='Do not sort images by hashed image ID.')
#    parser.add_argument('--seed', default=None, type=int,
#                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                    help='Outputs stuff for scripts/compute_mask.py.')
parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                    help='Do not crop output masks with the predicted bounding box.')
parser.add_argument('--image', default=None, type=str,
                    help='A path to an image to use for display.')
parser.add_argument('--images', default=None, type=str,
                    help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
parser.add_argument('--dir', default=None, type=str, dest='image_dir')
parser.add_argument('--video', default=None, type=str,
                    help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
parser.add_argument('--video_multiframe', default=1, type=int,
                    help='The number of frames to evaluate in parallel to make videos play at higher fps.')
parser.add_argument('--score_threshold', default=0, type=float,
                    help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                    help='Dont evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
parser.add_argument('--display_fps', default=True, dest='display_fps', action='store_true',
                    help='When displaying / saving video, draw the FPS on the frame')
parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                    help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')
parser.add_argument('--mask_crop', default=True, dest='mask_crop', help='Crop the image with mask')
parser.add_argument('--class_name', default=None, type=str, help='pick one class up like "person0"')
parser.add_argument('--transfer', default=True, dest='transfer', help='transfer the frame')

parser.add_argument("--vgg_dir", default='models/vgg_r31.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r31.pth',
                    help='pre-trained decoder path')
parser.add_argument("--style", default="data/style/25.jpg",
                    help='path to style image')  # sketch antimonocromatismo gogh #"data/style/gogh.jpg",
parser.add_argument("--matrixPath", default="models/r31.pth",
                    help='path to pre-trained model')
parser.add_argument('--fineSize', type=int, default=256,
                    help='crop image size')
parser.add_argument("--name", default="transferred_video",
                    help="name of generated video")
parser.add_argument("--layer", default="r31",
                    help="features of which layer to transfer")
parser.add_argument("--outf", default="real_time_demo_output",
                    help="output folder")

parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--total_epochs', type=int, default=10000)
parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
parser.add_argument('--train_n_batches', type=int, default=-1,
                    help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                    help="Spatial dimension to crop training samples for training")
parser.add_argument('--gradient_clip', type=float, default=None)
parser.add_argument('--schedule_lr_frequency', type=int, default=0, help='in number of iterations (0 for no schedule)')
parser.add_argument('--schedule_lr_fraction', type=float, default=10)
parser.add_argument("--rgb_max", type=float, default=255.)

parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
parser.add_argument('--no_cuda', action='store_true')

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
parser.add_argument('--validation_n_batches', type=int, default=-1)
parser.add_argument('--render_validation', action='store_true',
                    help='run inference (save flows to file) and every validation_frequency epoch')

parser.add_argument('--inference', action='store_true')
parser.add_argument('--inference_visualize', action='store_true',
                    help="visualize the optical flow during inference")
parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
                    help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
parser.add_argument('--inference_batch_size', type=int, default=1)
parser.add_argument('--inference_n_batches', type=int, default=-1)
parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

parser.add_argument('--skip_training', action='store_true')
parser.add_argument('--skip_validation', action='store_true')

parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--fp16_scale', type=float, default=1024.,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')

tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

tools.add_arguments_for_module(parser, torch.optim, argument_for_class='optimizer', default='Adam',
                               skip_params=['params'])

tools.add_arguments_for_module(parser, datasets, argument_for_class='training_dataset', default='MpiSintelFinal',
                               skip_params=['is_cropped'],
                               parameter_defaults={'root': './MPI-Sintel/flow/training'})

tools.add_arguments_for_module(parser, datasets, argument_for_class='validation_dataset', default='MpiSintelClean',
                               skip_params=['is_cropped'],
                               parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                   'replicates': 1})

tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean',
                               skip_params=['is_cropped'],
                               parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                   'replicates': 1})

parser.set_defaults(no_bar=False, display=False, resume_y=False, output_coco_json=False, output_web_json=False,
                    shuffle=False,
                    benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                    display_fps=False,
                    emulate_playback=False)

global args
args = parser.parse_args()

if args.output_web_json:
    args.output_coco_json = True

args.cuda = torch.cuda.is_available()
print(args)
os.makedirs(args.outf, exist_ok=True)
cudnn.benchmark = True

#    if args.seed is not None:
#        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {}  # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})
roi_save_n = 0


################# DATA #################
style = cv2.imread(args.style)/255
style = resize(style, (512,512, 3))
style = torch.Tensor(style).cuda().unsqueeze(0).permute(0,3,1,2)



################# MODEL #################
if (args.layer == 'r31'):
    matrix = MulLayer(layer='r31')
    vgg = encoder3()
    dec = decoder3()
elif (args.layer == 'r41'):
    matrix = MulLayer(layer='r41')
    vgg = encoder4()
    dec = decoder4()
vgg.load_state_dict(torch.load(args.vgg_dir))
dec.load_state_dict(torch.load(args.decoder_dir))
matrix.load_state_dict(torch.load(args.matrixPath))
for param in vgg.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False
for param in matrix.parameters():
    param.requires_grad = False

################# GLOBAL VARIABLE #################
content = torch.Tensor(1, 3, args.fineSize, args.fineSize)
cF = vgg(content)
################# GPU  #################
if (args.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()

    style = style.cuda()
    content = content.cuda()

totalTime = 0
imageCounter = 0
result_frames = []
contents = []
styles = []

with torch.no_grad():
    sF = vgg(style)

main_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(main_dir)

# Parse the official arguments
with tools.TimerBlock("Parsing Arguments") as block:
    if args.number_gpus < 0: args.number_gpus = torch.cuda.device_count()

    # Get argument defaults (hastag #thisisahack)
    parser.add_argument('--IGNORE', action='store_true')
    defaults = vars(parser.parse_args(['--IGNORE']))

    # Print all arguments, color the non-defaults
    for argument, value in sorted(vars(args).items()):
        reset = colorama.Style.RESET_ALL
        color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
        block.log('{}{}: {}{}'.format(color, argument, value, reset))

    args.model_class = tools.module_to_dict(models)[args.model]
    args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
    args.loss_class = tools.module_to_dict(losses)[args.loss]

    args.training_dataset_class = tools.module_to_dict(datasets)[args.training_dataset]
    args.validation_dataset_class = tools.module_to_dict(datasets)[args.validation_dataset]
    args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()
    args.log_file = join(args.save, 'args.txt')

    # dict to collect activation gradients (for training debug purpose)
    args.grads = {}

    if args.inference:
        args.skip_validation = True
        args.skip_training = True
        args.total_epochs = 1
        args.inference_dir = "{}/inference".format(args.save)

print('Source Code')
print(('  Current Git Hash: {}\n'.format(args.current_hash)))

# Change the title for `top` and `pkill` commands
setproctitle.setproctitle(args.save)

# Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments
with tools.TimerBlock("Initializing Datasets") as block:
    args.effective_batch_size = args.batch_size * args.number_gpus
    args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
    args.effective_number_workers = args.number_workers * args.number_gpus
    gpuargs = {'num_workers': args.effective_number_workers,
               'pin_memory': True,
               'drop_last': True} if args.cuda else {}
    inf_gpuargs = gpuargs.copy()
    inf_gpuargs['num_workers'] = args.number_workers

# Dynamically load model and loss class with parameters passed in via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
with tools.TimerBlock("Building {} model".format(args.model)) as block:
    class ModelAndLoss(nn.Module):
        def __init__(self, args):
            super(ModelAndLoss, self).__init__()
            kwargs = tools.kwargs_from_args(args, 'model')
            self.model = args.model_class(args, **kwargs)
            kwargs = tools.kwargs_from_args(args, 'loss')
            self.loss = args.loss_class(args, **kwargs)

        def forward(self, data, target, inference=False):
            output = self.model(data)

            loss_values = self.loss(output, target)

            if not inference:
                return loss_values
            else:
                return loss_values, output


    model_and_loss = ModelAndLoss(args)

    block.log('Effective Batch Size: {}'.format(args.effective_batch_size))
    block.log('Number of parameters: {}'.format(
        sum([p.data.nelement() if p.requires_grad else 0 for p in model_and_loss.parameters()])))

    # assing to cuda or wrap with dataparallel, model and loss
    if args.cuda and (args.number_gpus > 0) and args.fp16:
        block.log('Parallelizing')
        model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))

        block.log('Initializing CUDA')
        model_and_loss = model_and_loss.cuda().half()
        torch.cuda.manual_seed(args.seed)
        param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model_and_loss.parameters()]

    elif args.cuda and args.number_gpus > 0:
        block.log('Initializing CUDA')
        model_and_loss = model_and_loss.cuda()
        block.log('Parallelizing')
        model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))
        torch.cuda.manual_seed(args.seed)

    else:
        block.log('CUDA not being used')
        torch.manual_seed(args.seed)

    # Load weights if needed, otherwise randomly initialize
    if args.resume and os.path.isfile(args.resume):
        block.log("Loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        if not args.inference:
            args.start_epoch = checkpoint['epoch']
        best_err = checkpoint['best_EPE']
        model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
        block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))

    elif args.resume and args.inference:
        block.log("No checkpoint found at '{}'".format(args.resume))
        quit()

    else:
        block.log("Random initialization")

    block.log("Initializing save directory: {}".format(args.save))
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_logger = SummaryWriter(log_dir=os.path.join(args.save, 'train'), comment='training')
    validation_logger = SummaryWriter(log_dir=os.path.join(args.save, 'validation'), comment='validation')

# Dynamically load the optimizer with parameters passed in via "--optimizer_[param]=[value]" arguments
with tools.TimerBlock("Initializing {} Optimizer".format(args.optimizer)) as block:
    kwargs = tools.kwargs_from_args(args, 'optimizer')
    if args.fp16:
        optimizer = args.optimizer_class([p for p in param_copy if p.requires_grad], **kwargs)
    else:
        optimizer = args.optimizer_class([p for p in model_and_loss.parameters() if p.requires_grad], **kwargs)
    for param, default in list(kwargs.items()):
        block.log("{} = {} ({})".format(param, default, type(default)))

# Log all arguments to file
for argument, value in sorted(vars(args).items()):
    block.log2file(args.log_file, '{}: {}'.format(argument, value))


def opticalflow(args, frame1, frame2, warping_frame, model):

    model.eval()
    args.inference_n_batches = np.inf if args.inference_n_batches < 0 else args.inference_n_batches
    statistics = []
    total_loss = 0
    # frame2 = torch.Tensor(frame2).cuda()
    warping_frame = torch.Tensor(warping_frame).cuda()
    data = torch.zeros((1, 3, 2,) + frame1.size()[0:2]) #float32

    data[:, :, 0, ...] = frame1.permute(2, 0, 1).unsqueeze(0)
    data[:, :, 1, ...] = frame2.permute(2, 0, 1).unsqueeze(0)
    target = torch.zeros(data.size()[0:1] + (2,) + data.size()[-2:])
    # when ground-truth flows are not available for inference_dataset,
    # the targets are set to all zeros. thus, losses are actually L1 or L2 norms of compute optical flows,
    # depending on the type of loss norm passed in
    with torch.no_grad():
        losses, output = model(data, target, inference=True)  ##이것만 필요하지 않을까...


    losses = [torch.mean(loss_value) for loss_value in losses]
    loss_val = losses[0]  # Collect first loss for weight update
    total_loss += loss_val.item()
    loss_values = [v.item() for v in losses]

    statistics.append(loss_values)


    warping_frame = warping_frame.unsqueeze(0).permute(0, 3, 1, 2)
    B, C, H, W = warping_frame.shape
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    if warping_frame.is_cuda:
        grid = grid.cuda()

    vgrid = Variable(grid) + output * (-1)

 
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    output = nn.functional.grid_sample(warping_frame, vgrid, padding_mode='zeros')
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    return output


def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop,
                        score_threshold=args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    if args.transfer and num_dets_to_consider > 0:


        masks = masks[:num_dets_to_consider, :, :, None]

 
        for j in range(num_dets_to_consider):
            _class = cfg.dataset.class_names[classes[j]]
            if _class + str(j) == args.class_name:

                temp = img_gpu * masks[j]
                temp = temp[boxes[j, 1]:boxes[j, 3], boxes[j, 0]:boxes[j, 2], ...].cpu().numpy()
 
                temp = torch.Tensor(resize(temp, (512,512, 3))).cuda()
                temp = temp.permute(2, 0, 1).contiguous().unsqueeze(0)

                with torch.no_grad():
                    cF = vgg(temp)
                    if (args.layer == 'r41'):
                        feature, transmatrix = matrix(cF[args.layer], sF[args.layer])
                    else:
                        feature, transmatrix = matrix(cF, sF)
                    transfer = dec(feature)
                temp_img = transfer.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                temp_img = torch.Tensor(resize(temp_img, (boxes[j, 3] - boxes[j, 1], boxes[j,2] - boxes[j,0],3))).cuda()
                img_gpu = img_gpu * (1 - masks[j])  # + temp_img * masks[j]
                img_gpu[boxes[j, 1]:boxes[j, 1] + temp_img.shape[0], boxes[j, 0]:boxes[j, 0] + temp_img.shape[1], ...] \
                    += temp_img[...,] * masks[j][boxes[j, 1]:boxes[j, 1] + temp_img.shape[0],
                                  boxes[j, 0]:boxes[j, 0] + temp_img.shape[1]]
 
            else:
                continue

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_fps:
        # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if num_dets_to_consider == 0:
        return img_numpy
    return img_numpy


def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:args.top_k] for x in t]
        if isinstance(scores, list):
            box_scores = scores[0].cpu().numpy()
            mask_scores = scores[1].cpu().numpy()
        else:
            scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        boxes = boxes.cpu().numpy()
        masks = masks.cpu().numpy()

    with timer.env('Sync'):
        # Just in case
        torch.cuda.synchronize()


def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]


def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]


class Detections:

    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x) * 10) / 10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })

    def dump(self):
        dump_arguments = [
            (self.bbox_data, args.bbox_det_file),
            (self.mask_data, args.mask_det_file)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)

    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                       'use_yolo_regressors', 'use_prediction_matching',
                       'train_masks']

        output = {
            'info': {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)


def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()


def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()


def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections: Detections = None):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not args.output_coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))
            gt_masks = torch.Tensor(gt_masks).view(-1, h * w)

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes, gt_boxes = split(gt_boxes)
                crowd_masks, gt_masks = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop,
                                                    score_threshold=args.score_threshold)

        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores
        masks = masks.view(-1, h * w).cuda()
        boxes = boxes.cuda()

    if args.output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i, :], box_scores[i])
                    detections.add_mask(image_id, classes[i], masks[i, :, :], mask_scores[i])
            return

    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box', lambda i, j: bbox_iou_cache[i, j].item(),
             lambda i, j: crowd_bbox_iou_cache[i, j].item(),
             lambda i: box_scores[i], box_indices),
            ('mask', lambda i, j: mask_iou_cache[i, j].item(),
             lambda i, j: crowd_mask_iou_cache[i, j].item(),
             lambda i: mask_scores[i], mask_indices)
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])

        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)

                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue

                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue

                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j

                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue

                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)


def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = ((x >> 16) ^ x) & 0xFFFFFFFF
    return x


def evalimage(net: Yolact, path: str, save_path: str = None):
    #    if int(path[-8:-4])%5==0:
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)

    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]

    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)


def evalimages(net: Yolact, input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    for p in sorted(Path(input_folder).glob('*')):
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.jpg'
        out_path = os.path.join(output_folder, name)

        evalimage(net, path, out_path)
        print(path + ' -> ' + out_path)
    print('Done.')


from multiprocessing.pool import ThreadPool
from queue import Queue


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def evalvideo(net: Yolact, path: str, out_path: str = None):
    # If the path is a digit, parse it as a webcam index
    is_webcam = path.isdigit()

    # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
    cudnn.benchmark = True

    if is_webcam:
        vid = cv2.VideoCapture(int(path))
    else:
        vid = cv2.VideoCapture(path)

    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)

    target_fps = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_webcam:
        num_frames = float('inf')
    else:
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)
    fps = 0
    frame_time_target = 1 / target_fps
    running = True
    fps_str = ''
    vid_done = False
    frames_displayed = 0

    if out_path is not None:
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        if out_path is not None:
            out.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        frames = []
        for idx in range(args.video_multiframe):
            frame = vid.read()[1][:448, ...]  # optical flow크기 맞추기
            if frame is None:
                return frames
            frames.append(frame)
        return frames

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            num_extra = 0
            while imgs.size(0) < args.video_multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1
            out = net(imgs)
            if num_extra > 0:
                out = out[:-num_extra]

            return frames, out

    def prep_frame(inp, fps_str):
        with torch.no_grad():
            frame, preds = inp  ##frame이 이미지 preds에는 박스, 마스크, 클래스, 스코어 있음


            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)

    frame_buffer = Queue()

    video_fps = 0  ##여기서 오류나면 비디오 안켜지는거

    # All this timing code to make sure that
    def play_video():
        try:
            nonlocal frame_buffer, running, video_fps, is_webcam, num_frames, frames_displayed, vid_done

            video_frame_times = MovingAverage(100)
            frame_time_stabilizer = frame_time_target
            last_time = None
            stabilizer_step = 0.0005
            progress_bar = ProgressBar(30, num_frames)

            while running:
                frame_time_start = time.time()

                if not frame_buffer.empty():
                    next_time = time.time()
                    if last_time is not None:
                        video_frame_times.add(next_time - last_time)
                        video_fps = 1 / video_frame_times.get_avg()
                    if out_path is None:
                        cv2.imshow('transfer', frame_buffer.get().astype(np.uint8))  ###########result
                    else:
                        out.write(frame_buffer)
                    frames_displayed += 1
                    last_time = next_time

                    if out_path is not None:
                        if video_frame_times.get_avg() == 0:
                            fps = 0
                        else:
                            fps = 1 / video_frame_times.get_avg()
                        progress = frames_displayed / num_frames * 100
                        progress_bar.set_val(frames_displayed)

                        print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                              % (repr(progress_bar), frames_displayed, num_frames, progress, fps), end='')

                # This is split because you don't want savevideo to require cv2 display functionality (see #197)
                if out_path is None and cv2.waitKey(1) == 27:
                    # Press Escape to close
                    running = False
                if not (frames_displayed < num_frames):
                    running = False

                if not vid_done:
                    buffer_size = frame_buffer.qsize()
                    if buffer_size < args.video_multiframe:
                        frame_time_stabilizer += stabilizer_step
                    elif buffer_size > args.video_multiframe:
                        frame_time_stabilizer -= stabilizer_step
                        if frame_time_stabilizer < 0:
                            frame_time_stabilizer = 0

                    new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
                else:
                    new_target = frame_time_target

                next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
                target_time = frame_time_start + next_frame_target - 0.001  # Let's just subtract a millisecond to be safe

                if out_path is None or args.emulate_playback:
                    # This gives more accurate timing than if sleeping the whole amount at once
                    while time.time() < target_time:
                        time.sleep(0.001)
                else:
                    # Let's not starve the main thread, now
                    time.sleep(0.001)
        except:
            # See issue #197 for why this is necessary
            import traceback
            traceback.print_exc()

    extract_frame = lambda x, i: [
        x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]]]

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    first_batch = eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
    pool.apply_async(play_video)
    active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]
    original_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]
    # active_frames[0]['value'][0]
    if out_path is None: print('Press Escape to close.')
    try:
        while vid.isOpened() and running:
            # Hard limit on frames in buffer so we don't run out of memory >.>
            while frame_buffer.qsize() > 100:
                time.sleep(0.001)

            start_time = time.time()

            # Start loading the next frames from the disk
            if not vid_done:
                next_frames = pool.apply_async(get_next_frame, args=(vid,))
            else:
                next_frames = None

            if not (vid_done and len(active_frames) == 0):
 
                for i, frame in enumerate(active_frames):
                    _args = [frame['value']]
                    if frame['idx'] == 0:
                        _args.append(fps_str)
                        if i == 0:
                            frame['value'] = pool.apply_async(sequence[0], args=_args)
                        if i != 0:
                            frame['value'][0] = opticalflow(args=args, frame1=original_frames[i-1]['value'][0],
                                                         frame2=original_frames[i]['value'][0],
                                                         warping_frame=active_frames[0]['value'].get(),
                                                         model=model_and_loss)
                    else :
                        frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)
                for i, frame in enumerate(active_frames):


                    if frame['idx'] == 0 and i==0:
                        frame_buffer.put(frame['value'].get())
                    if frame['idx'] == 0 and i!=0:
                        frame_buffer.put(frame['value'][0])


                    # Remove the finished frames from the processing queue
                active_frames = [x for x in active_frames if x['idx'] > 0]
                original_frames = [x for x in original_frames if x['idx'] > 0]
                # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                for i, frame in enumerate(list(reversed(active_frames))):
                    if frame['idx'] == 0 and i!=0:
                        original_frames[i]['value'] = frame['value'][0]
                        frame['value'] = frame['value'][0]
                    else:
                        original_frames[i]['value'] = frame['value'].get()
                        frame['value'] = frame['value'].get()
                    frame['idx'] -= 1
                    original_frames[i]['idx'] -= 1

                    if frame['idx'] == 0:
                        original_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in
                                          range(1, len(frame['value'][0]))]
                        original_frames[i]['value'] = extract_frame(original_frames[i]['value'], 0)
                        # Split this up into individual threads for prep_frame since it doesn't support batch size
                        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in
                                          range(1, len(frame['value'][0]))]
                        frame['value'] = extract_frame(frame['value'], 0)

                # Finish loading in the next frames and add them to the processing queue
                if next_frames is not None:
                    frames = next_frames.get()
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence) - 1})
                        original_frames.append({'value': frames, 'idx': len(sequence) - 1})

                # Compute FPS
                frame_times.add(time.time() - start_time)
                fps = args.video_multiframe / frame_times.get_avg()
            else:
                fps = 0

            fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (
                fps, video_fps, frame_buffer.qsize())
            if not args.display_fps:
                print('\r' + fps_str + '    ', end='')

    except KeyboardInterrupt:
        print('\nStopping...')

    cleanup_and_exit()


def evaluate(net: Yolact, dataset, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    # TODO Currently we do not support Fast Mask Re-scroing in evalimage, evalimages, and evalvideo
    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(net, inp, out)
        else:
            evalimage(net, args.image)
        return
    elif args.images is not None:
        inp, out = args.images.split(':')
        evalimages(net, inp, out)
        return
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')
            evalvideo(net, inp, out)
        else:
            evalvideo(net, args.video)
        return

    frame_times = MovingAverage()
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)

    print()

    if not args.display and not args.benchmark:
        # For each class and iou, stores tuples (score, isPositive)
        # Index ap_data[type][iouIdx][classIdx]
        ap_data = {
            'box': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
            'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
        }
        detections = Detections()
    else:
        timer.disable('Load Data')

    dataset_indices = list(range(len(dataset)))

    if args.shuffle:
        random.shuffle(dataset_indices)
    elif not args.no_sort:
        # Do a deterministic shuffle based on the image ids
        #
        # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
        # the order of insertion. That means on python 3.6, the images come in the order they are in
        # in the annotations file. For some reason, the first images in the annotations file are
        # the hardest. To combat this, I use a hard-coded hash function based on the image ids
        # to shuffle the indices we use. That way, no matter what python version or how pycocotools
        # handles the data, we get the same result every time.
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])

    dataset_indices = dataset_indices[:dataset_size]

    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            timer.reset()

            with timer.env('Load Data'):
                img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)

                # Test flag, do not upvote
                if cfg.mask_proto_debug:
                    with open('scripts/info.txt', 'w') as f:
                        f.write(str(dataset.ids[image_idx]))
                    np.save('scripts/gt.npy', gt_masks)

                batch = Variable(img.unsqueeze(0))
                if args.cuda:
                    batch = batch.cuda()

            with timer.env('Network Extra'):
                preds = net(batch)
            # Perform the meat of the operation here depending on our mode.
            if args.display:
                img_numpy = prep_display(preds, img, h, w)
            elif args.benchmark:
                prep_benchmark(preds, h, w)
            else:
                prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)

            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            if it > 1:
                frame_times.add(timer.total_time())

            if args.display:
                if it > 1:
                    print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                plt.imshow(img_numpy)
                plt.title(str(dataset.ids[image_idx]))
                plt.show()
            elif not args.no_bar:
                if it > 1:
                    fps = 1 / frame_times.get_avg()
                else:
                    fps = 0
                progress = (it + 1) / dataset_size * 100
                progress_bar.set_val(it + 1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                      % (repr(progress_bar), it + 1, dataset_size, progress, fps), end='')

        if not args.display and not args.benchmark:
            print()
            if args.output_coco_json:
                print('Dumping detections...')
                if args.output_web_json:
                    detections.dump_web()
                else:
                    detections.dump()
            else:
                if not train_mode:
                    print('Saving data...')
                    with open(args.ap_data_file, 'wb') as f:
                        pickle.dump(ap_data, f)

                return calc_map(ap_data)
        elif args.benchmark:
            print()
            print()
            print('Stats for the last frame:')
            timer.print_stats()
            avg_seconds = frame_times.get_avg()
            print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000 * avg_seconds))

    except KeyboardInterrupt:
        print('Stopping...')


def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps


def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n: ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()


if args.config is not None:
    set_cfg(args.config)

if args.trained_model == 'interrupt':
    args.trained_model = SavePath.get_interrupt('weights/')
elif args.trained_model == 'latest':
    args.trained_model = SavePath.get_latest('weights/', cfg.name)

if args.config is None:
    model_path = SavePath.from_str(args.trained_model)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    args.config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % args.config)
    set_cfg(args.config)

if args.detect:
    cfg.eval_mask_branch = False

if args.dataset is not None:
    set_dataset(args.dataset)

with torch.no_grad():
    if not os.path.exists('results'):
        os.makedirs('results')

    if args.cuda:
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if args.resume_y and not args.display:
        with open(args.ap_data_file, 'rb') as f:
            ap_data = pickle.load(f)
        calc_map(ap_data)
        exit()

    if args.image is None and args.video is None and args.images is None:
        dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
        prep_coco_cats()
    else:
        dataset = None

    print('Loading model...', end='')
    net = Yolact()
    net.load_weights(args.trained_model)
    net.eval()
    print(' Done.')

    if args.cuda:
        net = net.cuda()

    evaluate(net, dataset)
