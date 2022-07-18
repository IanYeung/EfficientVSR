import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.archs.discriminator_arch import UNetDiscriminatorWithSpectralNorm


if __name__ == '__main__':
    device = torch.device('cuda')
    model_path = '/home/xiyang/Projects/BasicSR/experiments/pretrained_models/Discriminator/UNetDiscriminatorWithSpectralNorm.pth'
    # img_path = '/home/xiyang/Projects/BasicSR/results/BasicVSRGAN_30ResBlk_UNetDiscriminatorWithSpectralNorm_Vimeo90K_BDx4_FrameVar1e0/visualization/ToS3/room/00000149_BasicVSRGAN_30ResBlk_UNetDiscriminatorWithSpectralNorm_Vimeo90K_BDx4_FrameVar1e0.png'
    img_path = '/home/xiyang/Datasets/VSR-TEST-Renamed/ToS3/GT/room/00000149.png'
    img_np = cv2.imread(img_path) / 255.
    img_pt = img2tensor(img_np, bgr2rgb=True).unsqueeze(0).to(device)

    b, c, h, w = img_pt.size()
    croped_h, croped_w = h // 8 * 8, w // 8 * 8
    img_pt = img_pt[:, :, :croped_h, :croped_w]

    net = UNetDiscriminatorWithSpectralNorm(num_in_ch=3, num_feat=64)
    load_path = '/home/xiyang/Projects/BasicSR/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
    state_dict = torch.load(load_path, map_location=lambda storage, loc: storage)['params']
    net.load_state_dict(state_dict, strict=False)
    net = net.to(device)

    out_pt = net(img_pt)
    out_np = tensor2img(out_pt, min_max=(0, 1))
    # out_np = tensor2img(out_pt)

    # cv2.imwrite(filename='/home/xiyang/Results/Disc/disc_out_lq.png', img=out_np)
    cv2.imwrite(filename='/home/xiyang/Results/Disc/disc_out_gt.png', img=out_np)

