import os
import cv2
import glob
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.img_util import img2tensor, tensor2img


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def get_gaussian_kernel(kernel_size, sigma, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    xy_grid = xy_grid.float()
    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    return gaussian_kernel


def gaussian_smooth(x, kernel_size, sigma):
    kernel = get_gaussian_kernel(kernel_size, sigma, channels=1).to(x.device)
    with torch.no_grad():
        x = F.pad(x, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='reflect')
        x = F.conv2d(x, kernel, groups=1, padding=0)
    return x


def sigmoid_pt(x, scale=25, loc=0.12):
    return 1 / (1 + torch.exp(-scale * (x - loc)))


def eig_special_2d_pt(S):
    # B,3,H,W
    eps = 1e-6
    B, C, H, W = S.size()
    j11, j22, j12 = S[:, 0:1, :, :], S[:, 1:2, :, :], S[:, 2:3, :, :]
    term1 = j11 + j22
    term2 = ((j11 - j22) ** 2 + 4 * (j12 ** 2) + eps) ** 0.5
    val0 = (term1 + term2) / 2.0
    val1 = (term1 - term2) / 2.0
    val = torch.cat([val0, val1], 1)

    # Calcualte vec, y will be positive.
    vec0_0 = j12
    vec0_1 = (j22 - j11 + term2) / 2
    vec0 = torch.cat([vec0_0, vec0_1], 1)
    # Normalize.
    # vec0 = vec0/(eps+torch.sum(vec0*vec0,dim=1,keepdim=True)**0.5)
    # vec1 = vec1/(eps+torch.sum(vec1*vec1,dim=1,keepdim=True)**0.5)
    # Reshape and return.
    return val, vec0


def structure_mask(image, sigma, rho):
    image = torch.mean(image, 1, keepdim=True)
    structure_tensor = structure_tensor_2d_pt(image, sigma, rho)

    val, _ = eig_special_2d_pt(structure_tensor)
    straight_line = torch.abs(val[:, 0:1, :, :] - val[:, 1:2, :, :])
    edges = straight_line
    return edges


def structure_tensor_2d_pt(image, sigma, rho):
    # Make sure image is a gray tensor array.
    NR_kernel = int(max(np.ceil(sigma * 3) * 2 + 1, 3))
    kernel = get_gaussian_kernel(NR_kernel, sigma, channels=1).to(image.device)
    with torch.no_grad():
        image = F.pad(image, (NR_kernel // 2, NR_kernel // 2, NR_kernel // 2, NR_kernel // 2), mode='reflect')
        image = F.conv2d(image, kernel, groups=1, padding=0)
    # Compute derivatives (Scipy implementation truncates filter at 4 sigma).
    Iy_0 = image[:, :, 1:2, :] - image[:, :, 0:1, :]
    Iy_i = (image[:, :, 2:, :] - image[:, :, 0:-2, :]) / 2
    Iy_N = image[:, :, -1:, :] - image[:, :, -2:-1, :]
    Iy = torch.cat([Iy_0, Iy_i, Iy_N], 2)

    Ix_0 = image[:, :, :, 1:2] - image[:, :, :, 0:1]
    Ix_i = (image[:, :, :, 2:] - image[:, :, :, 0:-2]) / 2
    Ix_N = image[:, :, :, -1:] - image[:, :, :, -2:-1]
    Ix = torch.cat([Ix_0, Ix_i, Ix_N], 3)
    ##
    NR_kernel = int(max(np.ceil(rho * 3) * 2 + 1, 3))
    kernel = get_gaussian_kernel(NR_kernel, sigma, channels=1).to(image.device)
    with torch.no_grad():
        Ixx = Ix * Ix
        Ixx = F.pad(Ixx, (NR_kernel // 2, NR_kernel // 2, NR_kernel // 2, NR_kernel // 2), mode='reflect')
        Ixx = F.conv2d(Ixx, kernel, groups=1, padding=0)
        Iyy = Iy * Iy
        Iyy = F.pad(Iyy, (NR_kernel // 2, NR_kernel // 2, NR_kernel // 2, NR_kernel // 2), mode='reflect')
        Iyy = F.conv2d(Iyy, kernel, groups=1, padding=0)
        Ixy = Ix * Iy
        Ixy = F.pad(Ixy, (NR_kernel // 2, NR_kernel // 2, NR_kernel // 2, NR_kernel // 2), mode='reflect')
        Ixy = F.conv2d(Ixy, kernel, groups=1, padding=0)
    S = torch.cat([Ixx, Iyy, Ixy], 1)

    return S


def grad(image):
    Ix_0 = image[:, :, 1:2, :] - image[:, :, 0:1, :]
    Ix_i = (image[:, :, 2:, :] - image[:, :, 0:-2, :]) / 2
    Ix_N = image[:, :, -1:, :] - image[:, :, -2:-1, :]
    Ix = torch.cat([Ix_0, Ix_i, Ix_N], 2)

    Iy_0 = image[:, :, :, 1:2] - image[:, :, :, 0:1]
    Iy_i = (image[:, :, :, 2:] - image[:, :, :, 0:-2]) / 2
    Iy_N = image[:, :, :, -1:] - image[:, :, :, -2:-1]
    Iy = torch.cat([Iy_0, Iy_i, Iy_N], 3)

    image_grad = (Ix ** 2 + Iy ** 2) ** 0.5
    image_grad = torch.mean(image_grad, 1, keepdim=True)

    return image_grad


def get_mask2_nonor(x):
    # x = DoG_img(x,5,0.01)
    # x = torch.mean(x, dim=1, keepdim=True).reshape(x.shape[0], 1, -1).unsqueeze(3)
    # x_std = torch.std(x_gray,dim=2,unbiased=False,keepdim=True)  # B, 1, 1, 1
    scale = 2000
    GT_dog = torch.clamp((grad(x) ** 2) * scale, 0, 1)
    flat_region = 1 - gaussian_smooth(sigmoid_pt(GT_dog, scale=25, loc=0.1), 9, 2.0)
    # flat_region = flat_region * x_std*255/50
    edges_big = structure_mask(x, 0.1, 2) ** 2
    C = 5e2
    edges_big = C * edges_big
    edges_big = gaussian_smooth(edges_big, 9, 2.0)
    edges_big = torch.clamp(edges_big, 0.0, 1.0)
    edges_big = sigmoid_pt(edges_big, 25, 0.1)
    # edges_big = edges_big*x_std*255/50
    # mask = self.nor_pt(edges_big-0.5 + flat_region)
    mask = torch.clamp(edges_big + flat_region, 0.0, 1.0)
    # mask = self.sigmoid_pt(mask,10,0.5)
    return mask, edges_big, flat_region


if __name__ == '__main__':

    model_name = 'BasicVSRGAN_LDL_UNetDiscriminatorWithSpectralNorm_REDS_LDL5e2'

    gt_root = '/home/xiyang/Datasets/VSR-TEST/REDS4/test_sharp'
    sr_root = '/home/xiyang/Results/BasicSR/results/{}/visualization/REDS4'.format(model_name)

    save_root = '/home/xiyang/Results/REDS4/Mask'

    gt_file_paths = sorted(glob.glob(os.path.join(gt_root, '*', '*.png')))
    sr_file_paths = sorted(glob.glob(os.path.join(sr_root, '*', '*.png')))
    for file_path in list(zip(gt_file_paths, sr_file_paths)):
        gt_file_path, sr_file_path = file_path[0], file_path[1]

        tmp_gt = gt_file_path.split('/')
        tmp_sr = sr_file_path.split('/')

        assert tmp_gt[-2] == tmp_sr[-2]
        seq_name, frm_name = tmp_gt[-2], tmp_gt[-1]
        mkdir(os.path.join(save_root, seq_name))
        print('Processing sequcence {}/{}...'.format(seq_name, frm_name))

        gt_img = cv2.imread(gt_file_path) / 255.
        sr_img = cv2.imread(sr_file_path) / 255.

        # residual_np = np.sum(np.abs(gt_img - sr_img), axis=2, keepdims=True)
        # cv2.imwrite(filename=os.path.join(save_root, seq_name, frm_name), img=residual_np)

        # mask = cal_pixel_weight_local_global(img_target=img2tensor(gt_img).unsqueeze(0),
        #                                      img_output=img2tensor(sr_img).unsqueeze(0),
        #                                      ksize=7).squeeze(0)
        # mask_img = tensor2img(mask, min_max=(0, torch.max(mask)))
        # cv2.imwrite(filename=os.path.join(save_root, seq_name, frm_name), img=mask_img)

        mask, edge, flat = get_mask2_nonor(img2tensor(gt_img).unsqueeze(0))

        cv2.imwrite(filename=os.path.join(save_root, seq_name, f'mask_{frm_name}'), img=tensor2img(mask))
        cv2.imwrite(filename=os.path.join(save_root, seq_name, f'edge_{frm_name}'), img=tensor2img(edge))
        cv2.imwrite(filename=os.path.join(save_root, seq_name, f'flat_{frm_name}'), img=tensor2img(flat))


