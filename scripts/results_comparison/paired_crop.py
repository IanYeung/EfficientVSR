import os
import cv2
import numpy as np
from basicsr.utils import mkdir


def paired_crop(read_root, save_root, gt_root, models, mode,
                dataset, seq_idx, frm_idx, top_left, size):

    bottom_right = (top_left[0] + size, top_left[1] + size)

    # mkdir save root
    mkdir(os.path.join(save_root, mode, dataset, seq_idx))

    # GT
    gt_path = os.path.join(gt_root, dataset, 'GT', seq_idx, '{}.png'.format(frm_idx))
    gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    box_gt_img = cv2.rectangle(gt_img.copy(), top_left, bottom_right, (0, 0, 255), 3)
    crp_gt_img = gt_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
    cv2.imwrite(
        filename=os.path.join(save_root, mode, dataset, seq_idx, '{}_box_gt.png'.format(frm_idx)),
        img=box_gt_img
    )
    cv2.imwrite(
        filename=os.path.join(save_root, mode, dataset, seq_idx, '{}_crp_gt.png'.format(frm_idx)),
        img=crp_gt_img
    )

    # SR
    for model in models:
        img_path = os.path.join(read_root, model, 'visualization', dataset, seq_idx, '{}_{}.png'.format(frm_idx, model))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        crp_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
        cv2.imwrite(
            filename=os.path.join(save_root, mode, dataset, seq_idx, '{}_crp_{}.png'.format(frm_idx, model)),
            img=crp_img
        )


if __name__ == '__main__':

    # read_root = '/home/xiyang/Projects/BasicSR/results'
    # save_root = '/home/xiyang/Results/Presentation'
    # gt_root = '/home/xiyang/Datasets/VSR-TEST'
    # mode = 'degradation'
    # dataset, seq_idx, frm_idx, top_left, size = 'REDS4', '015', '00000008', (400, 150), 128
    # model_list = (
    #     'BasicVSRGAN_30ResBlk_UNetDiscriminatorWithSpectralNorm_REDS_BDx4',
    #     'BasicVSRGAN_30ResBlk_UNetDiscriminatorWithSpectralNorm_REDS_BIx4',
    # )
    # paired_crop(read_root, save_root, gt_root, model_list, mode, dataset, seq_idx, frm_idx, top_left, size)

    # read_root = '/home/xiyang/Projects/BasicSR/results'
    # save_root = '/home/xiyang/Results/Presentation'
    # gt_root = '/home/xiyang/Datasets/VSR-TEST'
    # mode = 'resblock'
    # dataset, seq_idx, frm_idx, top_left, size = 'REDS4', '015', '00000008', (400, 150), 128
    # model_list = (
    #     'BasicVSR_10ResBlk_REDS_BDx4',
    #     'BasicVSR_30ResBlk_REDS_BDx4',
    # )
    # paired_crop(read_root, save_root, gt_root, model_list, mode, dataset, seq_idx, frm_idx, top_left, size)

    # read_root = '/home/xiyang/Projects/BasicSR/results'
    # save_root = '/home/xiyang/Results/Presentation'
    # gt_root = '/home/xiyang/Datasets/VSR-TEST'
    # mode = 'model'
    # dataset, seq_idx, frm_idx, top_left, size = 'REDS4', '015', '00000010', (400, 150), 128
    # # dataset, seq_idx, frm_idx, top_left, size = 'REDS4', '011', '00000010', (100, 100), 128
    # model_list = (
    #     'BasicVSR_30ResBlk_REDS_BDx4',
    #     'BasicVSRCoupleProp_30ResBlk_REDS_BDx4',
    #     'BasicVSRPlusPlus_7ResBlk_REDS_BDx4'
    # )
    # paired_crop(read_root, save_root, gt_root, model_list, mode, dataset, seq_idx, frm_idx, top_left, size)

    read_root = '/home/xiyang/Projects/BasicSR/results'
    save_root = '/home/xiyang/Results/Presentation'
    gt_root = '/home/xiyang/Datasets/VSR-TEST'
    mode = 'discriminator'
    dataset, seq_idx, frm_idx, top_left, size = 'REDS4', '015', '00000008', (400, 150), 128
    model_list = (
        'BasicVSRGAN_30ResBlk_VGGStyleDiscriminator_REDSx4_BIx4',
        'BasicVSRGAN_30ResBlk_UNetDiscriminatorWithSpectralNorm_REDS_BIx4'
    )
    paired_crop(read_root, save_root, gt_root, model_list, mode, dataset, seq_idx, frm_idx, top_left, size)
