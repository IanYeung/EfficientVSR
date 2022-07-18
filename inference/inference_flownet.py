import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F

from basicsr.archs.spynet_arch import SpyNet
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.utils.flow_util import flow_to_image


def get_f_flow():
    device = torch.device('cuda')

    src_path = '/home/xiyang/Datasets/VSR-TEST/REDS4/Bicubic4xLR/000'
    dst_path = '/home/xiyang/Results/VSR/Plain/f_flow'
    load_path = '../experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'

    os.makedirs(os.path.join(dst_path), exist_ok=True)

    model = SpyNet(load_path=load_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(src_path, '*.png'))
        images = sorted(images, reverse=False)

        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            img1 = img2tensor(cv2.imread(imfile1, cv2.IMREAD_UNCHANGED) / 255., bgr2rgb=True, float32=True).to(device)
            img2 = img2tensor(cv2.imread(imfile2, cv2.IMREAD_UNCHANGED) / 255., bgr2rgb=True, float32=True).to(device)

            flow = model(img1.unsqueeze(0), img2.unsqueeze(0))
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            save_path = os.path.join(dst_path, '{}_flow.png'.format(os.path.basename(imfile1).split('.')[0]))
            cv2.imwrite(save_path, flow_to_image(flow))


def get_b_flow():
    device = torch.device('cuda')

    src_path = '/home/xiyang/Datasets/VSR-TEST/REDS4/Bicubic4xLR/000'
    dst_path = '/home/xiyang/Results/VSR/Plain/b_flow'
    load_path = '../experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'

    os.makedirs(os.path.join(dst_path), exist_ok=True)

    model = SpyNet(load_path=load_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(src_path, '*.png'))
        images = sorted(images, reverse=True)

        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            img1 = img2tensor(cv2.imread(imfile1, cv2.IMREAD_UNCHANGED) / 255., bgr2rgb=True, float32=True).to(device)
            img2 = img2tensor(cv2.imread(imfile2, cv2.IMREAD_UNCHANGED) / 255., bgr2rgb=True, float32=True).to(device)

            flow = model(img1.unsqueeze(0), img2.unsqueeze(0))
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            save_path = os.path.join(dst_path, '{}_flow.png'.format(os.path.basename(imfile1).split('.')[0]))
            cv2.imwrite(save_path, flow_to_image(flow))


if __name__ == '__main__':
    get_f_flow()
    get_b_flow()
