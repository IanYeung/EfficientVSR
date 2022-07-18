import os
import sys
import cv2
import glob
import json
import time
import shlex
import logging
import subprocess

from ffmpeg_quality_metrics import FfmpegQualityMetrics as ffqm


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def setup_logger(logger_name, log_file, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def calculate_metrics(src_video_root, dst_video_root, logger=None):

    src_video_paths = sorted(glob.glob(os.path.join(src_video_root, '*')))
    dst_video_paths = sorted(glob.glob(os.path.join(dst_video_root, '*')))

    assert len(src_video_paths) == len(dst_video_paths)

    avg_psnr_y, avg_ssim_y, avg_vmaf = 0., 0., 0.

    if logger:
        logger.info('-----------------------------------------------------------------')
        logger.info(src_video_root)
        logger.info(dst_video_root)
        logger.info('-----------------------------------------------------------------')
    else:
        print('-----------------------------------------------------------------')
        print(src_video_root)
        print(dst_video_root)
        print('-----------------------------------------------------------------')

    for idx, (src_video_path, dst_video_path) in enumerate(zip(src_video_paths, dst_video_paths)):
        # print('src path: {}'.format(src_video_path))
        # print('dst path: {}'.format(dst_video_path))

        results = ffqm(src_video_path, dst_video_path).calc(['ssim', 'psnr', 'vmaf'])

        psnr_y_list = [frm_results['psnr_y'] for frm_results in results['psnr']]
        ssim_y_list = [frm_results['ssim_y'] for frm_results in results['ssim']]
        vmaf_list = [frm_results['vmaf'] for frm_results in results['vmaf']]

        seq_avg_psnr_y = sum(psnr_y_list) / len(psnr_y_list)
        seq_avg_ssim_y = sum(ssim_y_list) / len(ssim_y_list)
        seq_avg_vmaf = sum(vmaf_list) / len(vmaf_list)

        avg_psnr_y += seq_avg_psnr_y / len(src_video_paths)
        avg_ssim_y += seq_avg_ssim_y / len(src_video_paths)
        avg_vmaf += seq_avg_vmaf / len(src_video_paths)

        if logger:
            logger.info('Sequence {:03d}:\t PNSR-Y: {:.2f}, SSIM-Y: {:.4f}, VMAF: {:.2f}'
                        .format(idx + 1, seq_avg_psnr_y, seq_avg_ssim_y, seq_avg_vmaf))
        else:
            print('Sequence {:03d}:\t PNSR-Y: {:.2f}, SSIM-Y: {:.4f}, VMAF: {:.2f}'
                  .format(idx + 1, seq_avg_psnr_y, seq_avg_ssim_y, seq_avg_vmaf))

    if logger:
        logger.info('-------------------------Average Results-------------------------')
        logger.info('PNSR-Y: {:.2f}, SSIM-Y: {:.4f}, VMAF: {:.2f}'.format(avg_psnr_y, avg_ssim_y, avg_vmaf))
    else:
        print('-------------------------Average Results-------------------------')
        print('PNSR-Y: {:.2f}, SSIM-Y: {:.4f}, VMAF: {:.2f}'.format(avg_psnr_y, avg_ssim_y, avg_vmaf))

    return avg_psnr_y, avg_ssim_y, avg_vmaf


def get_video_metadata(inp_video_path):
    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(inp_video_path)

    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobe_output = subprocess.check_output(args).decode('utf-8')
    ffprobe_output = json.loads(ffprobe_output)

    # # prints all the metadata available:
    # import pprint
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(ffprobe_output)
    #
    # h = ffprobe_output['streams'][0]['height']
    # w = ffprobe_output['streams'][0]['width']

    kbps = float(ffprobe_output['streams'][0]['bit_rate']) / 1000

    return kbps


def get_bitrate(src_root, logger=None):
    src_video_paths = sorted(glob.glob(os.path.join(src_root, '*.mp4')))
    avg_kbps = 0.

    if logger:
        logger.info('-----------------------------------------------------------------')
        logger.info(src_root)
        logger.info('-----------------------------------------------------------------')
    else:
        print('-----------------------------------------------------------------')
        print(src_root)
        print('-----------------------------------------------------------------')

    for src_video_path in src_video_paths:
        src_video_name = os.path.basename(src_video_path)
        kbps = get_video_metadata(src_video_path)
        avg_kbps += kbps / len(src_video_paths)
        if logger:
            logger.info('{}:\t {:.3f} kbps'.format(src_video_name, kbps))
        else:
            print('{}:\t {:.3f} kbps'.format(src_video_name, kbps))
    if logger:
        logger.info('------------------------------')
        logger.info('Average bitrate {:.3f} kbps'.format(avg_kbps))
    else:
        print('------------------------------')
        print('Average bitrate {:.3f} kbps'.format(avg_kbps))
    return avg_kbps


def save_bitrate_info(src_root, dst_root):
    mkdir(dst_root)
    src_video_paths = sorted(glob.glob(os.path.join(src_root, '*.mp4')))
    for src_video_path in src_video_paths:
        src_video_name = os.path.basename(src_video_path)
        dst_json_path = os.path.join(dst_root, src_video_name.replace('.mp4', '.json'))
        command = 'ffmpeg_bitrate_stats -a gop -of json {} > {}'.format(src_video_path, dst_json_path)
        os.system(command=command)


def evaluate(model, crf_list):
    # metrics
    log_path = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/results/metrics/{}.log'.format(model)
    setup_logger('metrics', log_path, level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('metrics')
    psnr_list, ssim_list, vmaf_list = list(), list(), list()
    for crf in crf_list:
        gt_video_root = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/videos-original/720P-MP4'
        lq_video_root = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/videos-compress/{}/CRF{}'.format(model, crf)
        psnr, ssim, vmaf = calculate_metrics(gt_video_root, lq_video_root, logger)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        vmaf_list.append(vmaf)
    print('PSNR list: ', psnr_list)
    print('SSIM list: ', ssim_list)
    print('VMAF list: ', vmaf_list)

    # bitrate
    log_path = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/results/bitrate/{}.log'.format(model)
    setup_logger('bitrate', log_path, level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('bitrate')
    kbps_list = list()
    for crf in crf_list:
        lq_video_root = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/videos-compress/{}/CRF{}'.format(model, crf)
        kbps = get_bitrate(lq_video_root, logger)
        kbps_list.append(kbps)
    print('PSNR list: ', kbps_list)


if __name__ == '__main__':

    # log_path = '/home/xiyang/data0/datasets/ReCp/MCL-JVC/results/metrics/{}.log'.format(model)
    # setup_logger('metrics', log_path, level=logging.INFO, screen=True, tofile=True)
    # logger = logging.getLogger('metrics')

    logger = None
    src_video_root = '/home/xiyang/Results/BasicSR/results/BasicVSRGAN_UNetDiscriminatorWithSpectralNorm_REDS_BDx4_LDL5e2/video/REDS4'
    dst_video_root = '/home/xiyang/Datasets/VSR-TEST/REDS4-video/test_sharp'
    psnr, ssim, vmaf = calculate_metrics(src_video_root, dst_video_root, logger=logger)


