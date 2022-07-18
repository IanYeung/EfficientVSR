import os
import cv2
import glob
import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils import img2tensor, tensor2img
from basicsr.archs import arch_util
from basicsr.archs import spynet_arch


def get_local_weights(batch_img, ksize):
    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    weighted_pix = torch.var(patches, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return weighted_pix


def cal_pixel_weight_local_global(img_target, img_output, ksize):
    diff_SR = torch.abs(img_target - img_output)
    diff_SR = torch.sum(diff_SR, 1, keepdim=True)
    weight_global = torch.var(diff_SR.clone(), dim=(-1, -2, -3), keepdim=True) ** 0.2
    pixel_weight = get_local_weights(diff_SR.clone(), ksize)
    overall_weight = weight_global * pixel_weight

    return overall_weight


def get_flow(x, flownet):
    b, n, c, h, w = x.size()

    x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
    x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

    flows_backward = flownet(x_1, x_2).view(b, n - 1, 2, h, w)
    flows_forward = flownet(x_2, x_1).view(b, n - 1, 2, h, w)

    return flows_forward, flows_backward


if __name__ == '__main__':

    device = torch.device('cuda:0')
    gt_root = '/home/xiyang/Datasets/VSR-TEST/REDS4/GT/000'
    sr_root = '/home/xiyang/Projects/BasicSR/results/BasicVSRGAN_UNetDiscriminatorWithSpectralNorm_REDS_BDx4/visualization/REDS4/000'
    save_root = '/home/xiyang/Results/test'

    gt_paths = sorted(glob.glob(os.path.join(gt_root, '*.png')))[:7]
    sr_paths = sorted(glob.glob(os.path.join(sr_root, '*.png')))[:7]

    gts, srs = [], []
    masks = []
    for idx, (gt_path, sr_path) in enumerate(list(zip(gt_paths, sr_paths))):
        gt = img2tensor(cv2.imread(gt_path) / 255.)
        sr = img2tensor(cv2.imread(sr_path) / 255.)
        gts.append(gt)
        srs.append(sr)

        mask = cal_pixel_weight_local_global(img_target=gt.unsqueeze(0),
                                             img_output=sr.unsqueeze(0),
                                             ksize=7).squeeze(0)
        masks.append(mask)
        # mask_img = tensor2img(mask, min_max=(0, torch.max(mask)))
        # cv2.imwrite(filename=os.path.join(save_root, f'mask_{idx:08d}.png'), img=mask_img)

    gts = torch.stack(gts, dim=0).unsqueeze(0).to(device)
    # srs = torch.stack(srs, dim=0).unsqueeze(0).to(device)
    masks_sin = torch.stack(masks, dim=0).unsqueeze(0).to(device)

    for i in range(masks_sin.size(1)):
        out = tensor2img(masks_sin[:, i, :, :, :], min_max=(0, 0.1))
        cv2.imwrite(filename=os.path.join(save_root, f'mask_sin_{i:08d}.png'), img=out)

    spynet_path = '/home/xiyang/Projects/BasicSR/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
    spynet = spynet_arch.SpyNet(spynet_path).to(device)

    flow_forward, flow_backward = get_flow(gts, spynet)

    wf_f, wf_b = arch_util.flow_warp_sequence_v2(masks_sin, flow_forward, flow_backward)

    # diff_sum = 0
    # for i in range(6):
    #     # out = tensor2img(wf_f[:, i, :, :, :])
    #     # cv2.imwrite(filename=os.path.join(save_root, f'wf_{i:08d}.png'), img=out)
    #
    #     diff = np.abs(tensor2img(wf_f[:, i, :, :, :]) - tensor2img(inp[:, i, :, :, :]))
    #     cv2.imwrite(filename=os.path.join(save_root, f'diff_v2_{i:08d}.png'), img=np.mean(diff, axis=2))
    #     diff_sum += diff
    #
    # print(np.sum(diff_sum) / (720 * 1280 * 6))
    # for i in range(6):
    #     out = tensor2img(wf_b[:, i, :, :, :])
    #     cv2.imwrite(filename=os.path.join(save_root, f'wb_{i:08d}.png'), img=out)

    masks_1 = masks_sin.clone()
    masks_1[:,  1:, :, :, :] = wf_f
    masks_2 = masks_sin.clone()
    masks_2[:, :-1, :, :, :] = wf_b
    masks_mul = (masks_sin + masks_1 + masks_2) / 3.

    for i in range(masks_mul.size(1)):
        out = tensor2img(masks_mul[:, i, :, :, :], min_max=(0, 0.1))
        cv2.imwrite(filename=os.path.join(save_root, f'mask_mul_{i:08d}.png'), img=out)
