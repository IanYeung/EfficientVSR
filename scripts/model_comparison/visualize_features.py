import argparse
import cv2
import glob
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img
from basicsr.utils.flow_util import flow_to_image
from basicsr.archs.visualize_arch import \
    BasicUniVSRFeatPropWithSpyFlow_Fast, \
    BasicUniVSRFeatPropWithPWCFlow_Fast, \
    BasicUniVSRFeatPropWithFastFlow_Fast, \
    BasicUniVSRFeatPropWithSpyFlowDCN_Fast, \
    BasicUniVSRFeatPropWithSpyFlowDCN_Fast_V2, \
    RealTimeBasicVSRCouplePropWithSpyNet, \
    RealTimeBasicVSRCouplePropWithPWCNet, \
    RealTimeBasicVSRCouplePropWithSpyNetDCN


def inference_unidirection(imgs, imgnames, model, save_path):
    with torch.no_grad():
        outputs, feat_currs, feat_props, feat_flows = model(imgs)

    feat_flows = feat_flows.squeeze()
    feat_flows = list(feat_flows)
    # outputs = outputs.squeeze()
    # outputs = list(outputs)

    os.makedirs(os.path.join(save_path, 'imgs'), exist_ok=True)
    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, 'imgs', f'{imgname}.png'), output)

    os.makedirs(os.path.join(save_path, 'flow'), exist_ok=True)
    for flow, imgname in zip(feat_flows, imgnames):
        flow = flow.permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(os.path.join(save_path, 'flow', f'{imgname}_flow.png'), flow_to_image(flow)[:, :, [2, 1, 0]])

    os.makedirs(os.path.join(save_path, 'feat_props'), exist_ok=True)
    for feat_prop, imgname in zip(feat_props, imgnames):
        feat_prop = feat_prop.squeeze(0).mean(0).cpu().numpy()
        feat_prop = ((feat_prop - np.min(feat_prop)) / (np.max(feat_prop) - np.min(feat_prop)) * 255.0).round()
        cv2.imwrite(os.path.join(save_path, 'feat_props', f'{imgname}.png'), feat_prop)

    os.makedirs(os.path.join(save_path, 'feat_currs'), exist_ok=True)
    for feat_curr, imgname in zip(feat_currs, imgnames):
        feat_curr = feat_curr.squeeze(0).mean(0).cpu().numpy()
        feat_curr = ((feat_curr - np.min(feat_curr)) / (np.max(feat_curr) - np.min(feat_curr)) * 255.0).round()
        cv2.imwrite(os.path.join(save_path, 'feat_currs', f'{imgname}.png'), feat_curr)

    return outputs, feat_currs, feat_props, feat_flows


def inference_bidirection(imgs, imgnames, model, save_path):
    with torch.no_grad():
        outputs, f_flows, b_flows = model(imgs)

    f_flows = list(f_flows.squeeze())
    b_flows = list(b_flows.squeeze())
    # outputs = list(outputs.squeeze())

    os.makedirs(os.path.join(save_path, 'imgs'), exist_ok=True)
    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, 'imgs', f'{imgname}.png'), output)

    os.makedirs(os.path.join(save_path, 'f_flow'), exist_ok=True)
    for flow, imgname in zip(f_flows, imgnames):
        flow = flow.permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(os.path.join(save_path, 'f_flow', f'{imgname}_flow.png'), flow_to_image(flow)[:, :, [2, 1, 0]])

    os.makedirs(os.path.join(save_path, 'b_flow'), exist_ok=True)
    for flow, imgname in zip(b_flows, imgnames):
        flow = flow.permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(os.path.join(save_path, 'b_flow', f'{imgname}_flow.png'), flow_to_image(flow)[:, :, [2, 1, 0]])

    return outputs, f_flows, b_flows


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'BasicUniVSRFeatPropWithSpyFlowDCN_20BLK_REDS_BIx4_V2'
    model_path = '../../experiments/pretrained_models/OnlineVSR/{}.pth'.format(model_name)
    input_path = '/home/xiyang/Datasets/VSR-TEST/REDS4/Bicubic4xLR/000'
    save_path = '/home/xiyang/Results/VSR/{}'.format(model_name)
    interval = 100

    # model = BasicUniVSRFeatPropWithSpyFlow_Fast(num_feat=64, num_extract_block=10, num_block=10)
    model = BasicUniVSRFeatPropWithSpyFlowDCN_Fast_V2(num_feat=64, num_extract_block=0, num_block=20)

    # model = BasicUniVSRFeatPropWithPWCFlow_Fast(num_feat=64, num_extract_block=0, num_block=20)
    # model = BasicUniVSRFeatPropWithFastFlow_Fast(num_feat=64, num_extract_block=0, num_block=20)
    # model = BasicUniVSRFeatPropWithSpyFlowDCN_Fast(num_feat=64, num_extract_block=0, num_block=20)
    # model = BasicUniVSRFeatPropWithSpyFlow_Fast_NoResLearn(num_feat=64, num_extract_block=0, num_block=20)
    # model = RealTimeBasicVSRCouplePropWithSpyNetDCN(num_feat=64, num_block=8)

    model.load_state_dict(torch.load(model_path)['params_ema'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(save_path, exist_ok=True)

    # load data and inference
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    num_imgs = len(imgs_list)
    if len(imgs_list) <= interval:  # too many images may cause CUDA out of memory
        imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
        imgs = imgs.unsqueeze(0).to(device)
        outputs, feat_currs, feat_propss, feat_flows = inference_unidirection(imgs, imgnames, model, save_path)
    else:
        for idx in range(0, num_imgs, interval):
            interval = min(interval, num_imgs - idx)
            imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            outputs, feat_currs, feat_props, feat_flows = inference_unidirection(imgs, imgnames, model, save_path)

    # # load data and inference
    # imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    # num_imgs = len(imgs_list)
    # if len(imgs_list) <= interval:  # too many images may cause CUDA out of memory
    #     imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
    #     imgs = imgs.unsqueeze(0).to(device)
    #     outputs, f_flows, b_flows = inference_bidirection(imgs, imgnames, model, save_path)
    # else:
    #     for idx in range(0, num_imgs, interval):
    #         interval = min(interval, num_imgs - idx)
    #         imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
    #         imgs = imgs.unsqueeze(0).to(device)
    #         outputs, f_flows, b_flows = inference_bidirection(imgs, imgnames, model, save_path)
