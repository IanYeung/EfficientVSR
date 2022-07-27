import argparse
import cv2
import glob
import os
import shutil
import torch
import numpy as np

from basicsr.archs.online_vsr_arch import \
    BasicUniVSRFeatPropWithoutAlign_Fast, \
    BasicUniVSRFeatPropWithSpyFlow_Fast, \
    BasicUniVSRFeatPropWithFastFlow_Fast, \
    BasicUniVSRFeatPropWithMaskFlow_Fast, \
    BasicUniVSRFeatPropWithSpyFlowDCN_Fast, \
    BasicUniVSRFeatPropWithFastFlowDCN_Fast, \
    BasicUniVSRFeatPropWithMaskFlowDCN_Fast, \
    BasicUniVSRFeatPropWithPCDAlign, \
    BasicUniVSRFeatPropWithFastFlowDCNWithDiff
from basicsr.archs.online_inference_arch import BasicUniVSRFeatPropWithFastFlowDCN_Test
from basicsr.archs.basicvsr_arch import \
    BasicVSR, BasicVSR_DirectUp, BasicVSR_DirectUp_NoFlow, \
    RealTimeBasicVSRCouplePropV2, RealTimeBasicVSRCouplePropV3, \
    RealTimeBasicVSRCouplePropV2_FF, RealTimeBasicVSRCouplePropV3_FF, \
    RealTimeBasicVSRCoupleProp_NoFlow
from basicsr.archs.efficient_vsr_arch import RLSP, RRN
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import img2tensor, tensor2img


def inference(inputs, imgnames, model, save_path):
    with torch.no_grad():
        outputs = model(inputs)
    # save imgs
    outputs = outputs.squeeze()
    outputs = list(outputs)
    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}.png'), output)


def model_inference():

    # model_name = 'BasicUniVSRFeatPropWithoutAlign_Fast_wiOneStageUp_30BLK32CHN_Vimeo90K_BDx4'
    # model_name = 'BasicUniVSRFeatPropWithFastFlowDCN_Fast_wiOneStageUp_10BLK_Vimeo90K_BDx4'
    # model_name = 'RLSP_62LAY32CHN_Vimeo90K_BDx4'
    # model_name = 'RRN_30BLK32CHN_Vimeo90K_BDx4'

    # model_name = 'BasicVSR_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'BasicVSR_DirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'BasicVSR_DirectUp_NoFlow_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'BasicUniVSRFeatPropWithPCDAlign_wiOneStageUp_10BLK32CHN_Vimeo90K_BDx4'

    model_name = 'BasicUniVSRFeatPropWithFastFlowDCN_Fast_FineFlow+6Conv_wiOneStageUp_10BLK32CHN_Vimeo90K_BDx4'

    # model_name = 'RealTimeBasicVSRCouplePropV2_WiDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCouplePropV2_WoDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCouplePropV3_WiDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCouplePropV3_WoDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCoupleProp_NoFlow_WiDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCoupleProp_NoFlow_WoDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCouplePropV3_FF_SpyNet_WiDirectUp_32CHN05BLK_Vimeo90K_BDx4'

    model_path = '/home/xiyang/Models/EfficientVSR/{}/models/net_g_latest.pth'.format(model_name)

    lq_root = '/home/xiyang/Datasets/VSR-TEST/UHD20/Gaussian4xLR'
    save_root = '/home/xiyang/data1/results/UHD20/{}'.format(model_name)

    interval = 50

    device = torch.device('cuda')

    # set up model
    # model = BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_extract_block=0, num_block=30, one_stage_up=True)
    # model = BasicUniVSRFeatPropWithMaskFlow_Fast(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True)
    # model = BasicUniVSRFeatPropWithFastFlow_Fast(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True)
    # model = BasicUniVSRFeatPropWithSpyFlow_Fast(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True)
    # model = BasicUniVSRFeatPropWithMaskFlowDCN_Fast(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True)
    model = BasicUniVSRFeatPropWithFastFlowDCN_Fast(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True)
    # model = BasicUniVSRFeatPropWithSpyFlowDCN_Fast(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True)
    # model = RLSP(filters=32, state_dim=32, layers=62)
    # model = RRN(n_c=32, n_b=30)
    # model = BasicVSR(num_feat=32, num_block=10)
    # model = BasicVSR_DirectUp(num_feat=32, num_block=10)
    # model = BasicVSR_DirectUp_NoFlow(num_feat=32, num_block=10)
    # model = BasicUniVSRFeatPropWithPCDAlign(num_feat=32, num_block=10, one_stage_up=True)
    # model = RealTimeBasicVSRCouplePropV2(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True)
    # model = RealTimeBasicVSRCouplePropV2_FF(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True)
    # model = RealTimeBasicVSRCouplePropV2(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=False)
    # model = RealTimeBasicVSRCouplePropV3(num_feat=32, num_extract_block=0, num_block=5, one_stage_up=True)
    # model = RealTimeBasicVSRCouplePropV3_FF(num_feat=32, num_extract_block=0, num_block=5, one_stage_up=True)
    # model = RealTimeBasicVSRCouplePropV3(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=False)
    # model = RealTimeBasicVSRCoupleProp_NoFlow(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True)
    # model = RealTimeBasicVSRCoupleProp_NoFlow(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=False)

    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    sequence_paths = sorted(glob.glob(os.path.join(lq_root, '*')))

    for sequence_path in sequence_paths:
        print('Processing sequence {}.'.format(os.path.basename(sequence_path)))
        save_path = os.path.join(save_root, os.path.basename(sequence_path))
        os.makedirs(save_path, exist_ok=True)
        # load data and inference
        imgs_list = sorted(glob.glob(os.path.join(sequence_path, '*')))
        num_imgs = len(imgs_list)
        if len(imgs_list) <= interval:  # too many images may cause CUDA out of memory
            imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            inference(imgs, imgnames, model, save_path)
        else:
            for idx in range(0, num_imgs, interval):
                interval = min(interval, num_imgs - idx)
                imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
                imgs = imgs.unsqueeze(0).to(device)
                inference(imgs, imgnames, model, save_path)


def model_evaluation():
    # model_name = 'RLSP_62LAY32CHN_Vimeo90K_BDx4'
    # model_name = 'RRN_30BLK32CHN_Vimeo90K_BDx4'
    # model_name = 'BasicVSR_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'BasicVSR_DirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'BasicVSR_DirectUp_NoFlow_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'BasicUniVSRFeatPropWithPCDAlign_wiOneStageUp_10BLK32CHN_Vimeo90K_BDx4'
    # model_name = 'BasicUniVSRFeatPropWithMaskFlow_Fast_wiOneStageUp_10BLK32CHN_Vimeo90K_BDx4'
    # model_name = 'BasicUniVSRFeatPropWithMaskFlowDCN_Fast_wiOneStageUp_10BLK32CHN_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCouplePropV2_WiDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCouplePropV2_WoDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCouplePropV3_WiDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCouplePropV3_WoDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCoupleProp_NoFlow_WiDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCoupleProp_NoFlow_WoDirectUp_32CHN10BLK_Vimeo90K_BDx4'
    model_name = 'BasicUniVSRFeatPropWithFastFlowDCN_Fast_FineFlow+6Conv_wiOneStageUp_10BLK32CHN_Vimeo90K_BDx4'
    # model_name = 'RealTimeBasicVSRCouplePropV3_SpyNet_WiDirectUp_32CHN05BLK_Vimeo90K_BDx4'

    gt_root = '/home/xiyang/Datasets/VSR-TEST/UHD20/GT'
    save_root = '/home/xiyang/data1/results/UHD20-Online/{}'.format(model_name)

    print(model_name)
    psnr_list, ssim_list = [], []
    seq_paths = sorted(glob.glob(os.path.join(save_root, '*')))
    for seq_path in seq_paths:
        seq_name = os.path.basename(seq_path)
        frm_paths = sorted(glob.glob(os.path.join(seq_path, '*.png')))
        psnr_avg, ssim_avg = 0., 0.
        for frm_path in frm_paths:
            frm_name = os.path.basename(frm_path)
            hq_img = cv2.imread(frm_path)
            gt_img = cv2.imread(os.path.join(gt_root, seq_name, frm_name))
            psnr_avg += calculate_psnr(hq_img, gt_img, 0)
            ssim_avg += calculate_ssim(hq_img, gt_img, 0)
        psnr_list.append(psnr_avg / len(frm_paths))
        ssim_list.append(ssim_avg / len(frm_paths))
        print('{}, PSNR: {:.2f}dB, SSIM: {:.4f}.'.format(seq_name, psnr_avg / len(frm_paths), ssim_avg / len(frm_paths)))
    print('Mean PSNR: {:.2f} dB.'.format(sum(psnr_list) / len(psnr_list)))
    print('Mean SSIM: {:.4f}.'.format(sum(ssim_list) / len(ssim_list)))


def online_inference():
    def read_img(path):
        img = cv2.imread(path).astype(np.float32) / 255.
        img = img2tensor(img, bgr2rgb=True, float32=True)[None, :]
        return img

    model_name = 'BasicUniVSRFeatPropWithFastFlowDCN_Fast_FineFlow+6Conv_wiOneStageUp_10BLK32CHN_Vimeo90K_BDx4'
    model_path = '/home/xiyang/Models/EfficientVSR/{}/models/net_g_latest.pth'.format(model_name)
    lq_root = '/home/xiyang/Datasets/VSR-TEST/UHD20/Gaussian4xLR'
    save_root = '/home/xiyang/data1/results/UHD20-Online/{}'.format(model_name)

    device = torch.device('cuda')

    model = BasicUniVSRFeatPropWithFastFlowDCN_Test(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True)
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    sequence_paths = sorted(glob.glob(os.path.join(lq_root, '*')))
    for sequence_path in sequence_paths:
        print('Processing sequence {}.'.format(os.path.basename(sequence_path)))
        save_path = os.path.join(save_root, os.path.basename(sequence_path))
        os.makedirs(save_path, exist_ok=True)
        # load data and inference
        imgs_list = sorted(glob.glob(os.path.join(sequence_path, '*')))
        num_imgs = len(imgs_list)

        frm_prev = read_img(imgs_list[0]).to(device)
        b, c, h, w = frm_prev.size()
        feat_prop = frm_prev.new_zeros(1, 32, h, w).to(device)

        for idx in range(len(imgs_list)):
            img_name = os.path.splitext(os.path.basename(imgs_list[idx]))[0]
            print('{}'.format(img_name))
            frm_curr = read_img(imgs_list[idx]).to(device)
            with torch.no_grad():
                init = idx == 0
                out_curr, feat_prop = model(frm_curr, frm_prev, feat_prop, init=init)
                frm_prev = frm_curr
            output = tensor2img(out_curr)
            cv2.imwrite(os.path.join(save_path, f'{img_name}.png'), output)


if __name__ == '__main__':
    # model_inference()
    model_evaluation()
    # online_inference()