from __future__ import print_function
import argparse
from math import log10
import numpy as np
import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from module_util import initialize_weights
from dataset import build_dataloader
import pdb
import socket
import time
import skimage
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io
from models import InpaintingModel

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Video Inpainting with Background Auxilary')
parser.add_argument('--img_flist', type=str, default='/home/lab265/lab265/liulu/edge_connect/datasets/celeba_test.flist')
parser.add_argument('--mask_flist', type=str, default='/home/lab265/lab265/csy/datasets/irregular_mask/celeba-10-20.flist')
parser.add_argument('--save', default=True, action='store_true', help='If save test images')
parser.add_argument('--save_path', type=str, default='./test_results/celeba-10-20-results')
parser.add_argument('--model', default='./checkpoints/save/x_1080tiRN_rn_bs_6_epoch_last.pth',
                    help='sr pretrained base model')
parser.add_argument('--bs', type=int, default=1, help='training batch size')


parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--cpu', default=False, action='store_true', help='Use CPU to test')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=67454, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--threshold', type=float, default=0.8)
parser.add_argument('--input_size', type=int, default=256, help='input image size')
parser.add_argument('--l1_weight', type=float, default=1.0)
parser.add_argument('--gan_weight', type=float, default=0.1)

opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def eval():
    model.eval()
    model.generator.eval()
    count = 1
    avg_du = 0
    avg_psnr, avg_ssim, avg_l1 = 0., 0., 0.
    for batch in testing_data_loader:
        gt, mask, index = batch
        t_io2 = time.time()
        if cuda:
            gt = gt.cuda()
            mask = mask.cuda()

        ## The test or ensemble test

        # t0 = time.clock()
        with torch.no_grad():
            prediction = model.generator(gt, mask)
            prediction = prediction * mask + gt * (1 - mask)
        # t1 = time.clock()
        # du = t1 - t0
        # print("===> Processing: %s || Timer: %.4f sec." % (str(count), du))

        # avg_du += du
        # print(
        #     "Number: %05d" % (count),
        #     " | Average time: %.4f" % (avg_du/count))

        # Save the video frames
        batch_avg_psnr, batch_avg_ssim, batch_avg_l1 = evaluate_batch(
            batch_size=opt.bs,
            gt_batch=gt,
            pred_batch=prediction,
            mask_batch=mask,
            save=opt.save,
            path=opt.save_path,
            count=count,
            index=index
        )

        # avg_psnr = (avg_psnr * (count - 1) + batch_avg_psnr) / count
        avg_psnr = avg_psnr + ((batch_avg_psnr - avg_psnr) / count)
        avg_ssim = avg_ssim + ((batch_avg_ssim - avg_ssim) / count)
        avg_l1 = avg_l1 + ((batch_avg_l1 - avg_l1) / count)

        print(
            "Number: %05d" % (count * opt.bs),
            " | Average: PSNR: %.4f" % (avg_psnr),
            " SSIM: %.4f" % (avg_ssim),
            " L1: %.4f" % (avg_l1),
            "| Current batch:", count,
            " PSNR: %.4f" % (batch_avg_psnr),
            " SSIM: %.4f" % (batch_avg_ssim),
            " L1: %.4f" % (batch_avg_l1), flush=True
        )

        count += 1


def save_img(path, name, img):
    # img (H,W,C) or (H,W) np.uint8
    skimage.io.imsave(path + '/' + name + '.png', img)


def PSNR(pred, gt, shave_border=0):
    return compare_psnr(pred, gt, data_range=255)
    # imdff = pred - gt
    # rmse = math.sqrt(np.mean(imdff ** 2))
    # if rmse == 0:
    #     return 100
    # return 20 * math.log10(255.0 / rmse)


def L1(pred, gt):
    return np.mean(np.abs((np.mean(pred, 2) - np.mean(gt, 2)) / 255))


def SSIM(pred, gt, data_range=255, win_size=11, multichannel=True):
    return compare_ssim(pred, gt, data_range=data_range,
                        multichannel=multichannel, win_size=win_size)


def evaluate_batch(batch_size, gt_batch, pred_batch, mask_batch, save=False, path=None, count=None, index=None):
    pred_batch = pred_batch * mask_batch + gt_batch * (1 - mask_batch)

    if save:
        input_batch = gt_batch * (1 - mask_batch) + mask_batch
        input_batch = (input_batch.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        mask_batch = (mask_batch.detach().permute(0, 2, 3, 1).cpu().numpy()[:, :, :, 0] * 255).astype(np.uint8)

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path + '/input'):
            os.mkdir(path + '/input')
        if not os.path.exists(path + '/mask'):
            os.mkdir(path + '/mask')
        if not os.path.exists(path + '/output'):
            os.mkdir(path + '/output')
        if not os.path.exists(path + '/gt'):
            os.mkdir(path + '/gt')

    gt_batch = (gt_batch.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    pred_batch = (pred_batch.detach().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    psnr, ssim, l1 = 0., 0., 0.

    for i in range(batch_size):
        gt, pred, name = gt_batch[i], pred_batch[i], index[i].data.item()

        psnr += PSNR(pred, gt)
        # print(pred)
        ssim += SSIM(pred, gt)
        l1 += L1(pred, gt)

        if save:
            save_img(path+'/input', str(name), input_batch[i])
            save_img(path+'/mask', str(name), mask_batch[i])
            save_img(path+'/output', str(name), pred_batch[i])
            save_img(path+'/gt', str(name), gt_batch[i])

    return psnr / batch_size, ssim / batch_size, l1 / batch_size


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


if __name__ == '__main__':
    if opt.cpu:
        print("===== Use CPU to Test! =====")
    else:
        print("===== Use GPU to Test! =====")

    ## Set the GPU mode
    gpus_list = range(opt.gpus)
    cuda = not opt.cpu
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # Model
    model = InpaintingModel(g_lr=opt.lr, d_lr=(0.1 * opt.lr), l1_weight=opt.l1_weight, gan_weight=opt.gan_weight,
                            iter=0, threshold=opt.threshold)
    print('---------- Networks architecture -------------')
    print("Generator:")
    print_network(model.generator)
    print("Discriminator:")
    print_network(model.discriminator)
    print('----------------------------------------------')

    pretained_model = torch.load(opt.model, map_location=lambda storage, loc: storage)

    if cuda:
        model = model.cuda()
        model.load_state_dict(pretained_model)
        # model.generator = torch.nn.DataParallel(model.generator, device_ids=gpus_list)
        # model.discriminator = torch.nn.DataParallel(model.discriminator, device_ids=gpus_list)
        # model.load_state_dict(pretained_model)
    else:
        new_state_dict = model.state_dict()
        for k, v in pretained_model.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    # pretained_G_model = torch.load(opt.model, map_location=lambda storage, loc: storage)
    # model.generator.load_state_dict(pretained_G_model)
    print('Pre-trained G model is loaded.')

    # Datasets
    print('===> Loading datasets')
    testing_data_loader = build_dataloader(
        flist=opt.img_flist,
        mask_flist=opt.mask_flist,
        augment=False,
        training=False,
        input_size=opt.input_size,
        batch_size=opt.bs,
        num_workers=opt.threads,
        shuffle=False
    )
    print('===> Loaded datasets')

    ## Eval Start!!!!
    eval()
