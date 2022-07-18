import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn as nn
from torch.nn import functional as F

from copy import deepcopy
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.utils.flow_util import flow_to_image
from basicsr.archs.arch_util import flow_warp, resize_flow
from basicsr.archs.spynet_arch import SpyNet


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


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    return img_flo


def compare_flow():
    dataset_root = '/home/xiyang/Datasets/VSR-TEST'
    project_root = '/home/xiyang/Projects'
    results_root = '/home/xiyang/Results'

    dataset = 'REDS4'
    mode = 'GT'
    seq_idx = '015'

    img1_path = os.path.join(dataset_root, f'{dataset}/{mode}/{seq_idx}/00000007.png')
    img2_path = os.path.join(dataset_root, f'{dataset}/{mode}/{seq_idx}/00000008.png')
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1_pt = img2tensor(img1 / 255.).unsqueeze(0)
    img2_pt = img2tensor(img2 / 255.).unsqueeze(0)
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'img_prev.png'), img=img1)
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'img_next.png'), img=img2)

    flownet = SpyNet()

    # Pre-trained flow
    load_path = os.path.join(project_root,
                             'BasicSR/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth')
    state_dict1 = torch.load(load_path, map_location=lambda storage, loc: storage)['params']
    flownet.load_state_dict(state_dict1, strict=False)

    flo = flownet(img1_pt, img2_pt)
    img_flo = flow_to_image(flo[0].permute(1, 2, 0).detach().cpu().numpy())
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'flow_pretrained.png'), img=img_flo[:, :, [2, 1, 0]])
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # Fine-tuned flow
    load_path = os.path.join(project_root,
                             'BasicSR/experiments/pretrained_models/BasicVSR/BasicVSR_Vimeo90K_BDx4-e9bf46eb.pth')
    state_dict2 = torch.load(load_path, map_location=lambda storage, loc: storage)['params']
    for k, v in deepcopy(state_dict2).items():
        if not k.startswith('spynet.'):
            state_dict2.pop(k)
    for k, v in deepcopy(state_dict2).items():
        if k.startswith('spynet.'):
            state_dict2[k[7:]] = v
            state_dict2.pop(k)
    flownet.load_state_dict(state_dict2, strict=True)

    flo = flownet(img1_pt, img2_pt)
    img_flo = flow_to_image(flo[0].permute(1, 2, 0).detach().cpu().numpy())
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'flow_trained.png'), img=img_flo[:, :, [2, 1, 0]])
    # plt.imshow(img_flo / 255.0)
    # plt.show()


if __name__ == '__main__':

    model_name = 'BasicVSRGAN_30ResBlk_UNetDiscriminatorWithSpectralNorm_Vimeo90K_BDx4'
    gt_data_root = '/home/xiyang/Datasets/VSR-TEST-Renamed'
    lq_data_root = '/home/xiyang/Projects/BasicSR/results/{}/visualization'.format(model_name)
    project_root = '/home/xiyang/Projects'
    results_root = '/home/xiyang/Results'

    dataset, mode, seq_idx, frm_idx = 'REDS4', 'GT', '000', 1
    img1_gt_path = os.path.join(gt_data_root, f'{dataset}/{mode}/{seq_idx}/{frm_idx+0:08d}.png')
    img2_gt_path = os.path.join(gt_data_root, f'{dataset}/{mode}/{seq_idx}/{frm_idx+1:08d}.png')
    img1_gt = cv2.imread(img1_gt_path)
    img2_gt = cv2.imread(img2_gt_path)
    img1_gt_pt = img2tensor(img1_gt / 255.).unsqueeze(0)
    img2_gt_pt = img2tensor(img2_gt / 255.).unsqueeze(0)
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'gt_img_curr.png'), img=img1_gt)
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'gt_img_next.png'), img=img2_gt)

    img1_lq_path = os.path.join(lq_data_root, f'{dataset}/{seq_idx}/{frm_idx+0:08d}_{model_name}.png')
    img2_lq_path = os.path.join(lq_data_root, f'{dataset}/{seq_idx}/{frm_idx+1:08d}_{model_name}.png')
    img1_lq = cv2.imread(img1_lq_path)
    img2_lq = cv2.imread(img2_lq_path)
    img1_lq_pt = img2tensor(img1_lq / 255.).unsqueeze(0)
    img2_lq_pt = img2tensor(img2_lq / 255.).unsqueeze(0)
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'lq_img_curr.png'), img=img1_lq)
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'lq_img_next.png'), img=img2_lq)

    flownet = SpyNet()

    # Pre-trained flow
    load_path = os.path.join(project_root, 'BasicSR/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth')
    state_dict = torch.load(load_path, map_location=lambda storage, loc: storage)['params']
    flownet.load_state_dict(state_dict, strict=False)

    flow = flownet(img1_gt_pt, img2_gt_pt)
    img_flow = flow_to_image(flow[0].permute(1, 2, 0).detach().cpu().numpy())
    # cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'flow.png'), img=img_flow[:, :, [2, 1, 0]])

    img2_gt_pt_warped = flow_warp(img2_gt_pt, flow.permute(0, 2, 3, 1))
    img2_gt_warped = tensor2img(img2_gt_pt_warped)
    residual_gt = np.mean(np.abs(img1_gt - img2_gt_warped), axis=2)
    print(np.mean(residual_gt))
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'gt_img_next_warped.png'), img=img2_gt_warped)
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'residual_gt.png'), img=residual_gt)

    img2_lq_pt_warped = flow_warp(img2_lq_pt, flow.permute(0, 2, 3, 1))
    img2_lq_warped = tensor2img(img2_lq_pt_warped)
    residual_lq = np.mean(np.abs(img1_lq - img2_lq_warped), axis=2)
    print(np.mean(residual_lq))
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'lq_img_next_warped.png'), img=img2_lq_warped)
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'residual_lq.png'), img=residual_lq)

    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'residual.png'), img=residual_gt - residual_lq)

    residual_gt_pt = torch.from_numpy(residual_gt / 255.).unsqueeze(0)
    residual_lq_pt = torch.from_numpy(residual_lq / 255.).unsqueeze(0)

    # # mask 3
    # mask = cal_pixel_weight_local_global(img_target=residual_gt_pt.unsqueeze(0),
    #                                      img_output=residual_lq_pt.unsqueeze(0),
    #                                      ksize=3).squeeze(0)
    # mask_img = tensor2img(mask, min_max=(torch.min(mask), torch.max(mask)))
    # cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'mask-3.png'), img=mask_img)
    # # mask 5
    # mask = cal_pixel_weight_local_global(img_target=residual_gt_pt.unsqueeze(0),
    #                                      img_output=residual_lq_pt.unsqueeze(0),
    #                                      ksize=5).squeeze(0)
    # mask_img = tensor2img(mask, min_max=(torch.min(mask), torch.max(mask)))
    # cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'mask-5.png'), img=mask_img)
    # # mask 7
    # mask = cal_pixel_weight_local_global(img_target=residual_gt_pt.unsqueeze(0),
    #                                      img_output=residual_lq_pt.unsqueeze(0),
    #                                      ksize=7).squeeze(0)
    # mask_img = tensor2img(mask, min_max=(torch.min(mask), torch.max(mask)))
    # cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'mask-7.png'), img=mask_img)
    # # mask 9
    # mask = cal_pixel_weight_local_global(img_target=residual_gt_pt.unsqueeze(0),
    #                                      img_output=residual_lq_pt.unsqueeze(0),
    #                                      ksize=9).squeeze(0)
    # mask_img = tensor2img(mask, min_max=(torch.min(mask), torch.max(mask)))
    # cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'mask-9.png'), img=mask_img)

    # frame variation
    frame_var_gt = np.mean(np.abs(img1_gt - img2_gt), axis=2)
    frame_var_lq = np.mean(np.abs(img1_lq - img2_lq), axis=2)
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'frame_var_gt.png'), img=frame_var_gt)
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'frame_var_lq.png'), img=frame_var_lq)

    frame_var_res = np.abs(frame_var_gt - frame_var_lq)
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'frame_var_res.png'), img=frame_var_res)

    frame_var_gt_pt = img1_gt_pt - img2_gt_pt
    frame_var_lq_pt = img1_lq_pt - img2_lq_pt
    # mask 3
    mask3 = cal_pixel_weight_local_global(img_target=frame_var_gt_pt,
                                         img_output=frame_var_lq_pt,
                                         ksize=3).squeeze(0)
    mask_img = tensor2img(mask3, min_max=(torch.min(mask3), torch.max(mask3)))
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'frame-var-mask-3.png'), img=mask_img)
    # mask 5
    mask5 = cal_pixel_weight_local_global(img_target=frame_var_gt_pt,
                                         img_output=frame_var_lq_pt,
                                         ksize=5).squeeze(0)
    mask_img = tensor2img(mask5, min_max=(torch.min(mask5), torch.max(mask5)))
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'frame-var-mask-5.png'), img=mask_img)
    # mask 7
    mask7 = cal_pixel_weight_local_global(img_target=frame_var_gt_pt,
                                         img_output=frame_var_lq_pt,
                                         ksize=7).squeeze(0)
    mask_img = tensor2img(mask7, min_max=(torch.min(mask7), torch.max(mask7)))
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'frame-var-mask-7.png'), img=mask_img)
    # mask 9
    mask9 = cal_pixel_weight_local_global(img_target=frame_var_gt_pt,
                                         img_output=frame_var_lq_pt,
                                         ksize=9).squeeze(0)
    mask_img = tensor2img(mask9, min_max=(torch.min(mask9), torch.max(mask9)))
    cv2.imwrite(filename=os.path.join(results_root, 'Flow', 'frame-var-mask-9.png'), img=mask_img)
