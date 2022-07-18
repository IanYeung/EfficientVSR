import os
import os.path as osp
import cv2
import glob
import ffmpeg
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def comisr_encode_decode():
    command = f'ffmpeg -framerate 10 -i im%2d.png -c:v libx264 -crf 0 lossless.mp4 ' \
              f'&& ffmpeg -i lossless.mp4 -vcodec libx264 -crf 25 crf25.mp4 ' \
              f'&& ffmpeg -ss 00:00:00 -t 00:00:10 -i crf25.mp4 -r 10 crf25_%2d.png'
    print(command)
    os.system(command)


def encode_video_with_ffmpeg(src_path, dst_path, crf=0, fps=25, lib='libx264'):
    command = 'ffmpeg -r {} -f image2 -i {} -c:v {} -crf {} -pix_fmt yuv420p -an {} -y'\
        .format(fps, src_path, lib, crf, dst_path)
    print(command)
    os.system(command)


def generate_video_cfgs(raw_video_dir, cfg_save_dir):
    raw_video_list = sorted(glob.glob(os.path.join(raw_video_dir, "*.yuv")))
    num_videos = len(raw_video_list)
    print(f'{num_videos} videos found.')

    for ite_vid in range(num_videos):
        raw_video_path = raw_video_list[ite_vid]
        raw_video_name = os.path.basename(raw_video_path).split(".")[0]
        # _res = raw_video_name.split("_")[1]
        # w = _res.split("x")[0]
        # h = _res.split("x")[1]
        # nfs = raw_video_name.split("_")[2]

        # _res = raw_video_name.split("_")[1]
        w = '480'
        h = '272'
        nfs = '64'

        cfg_path = os.path.join(cfg_save_dir, raw_video_name + ".cfg")
        fp = open(cfg_path, 'w')

        _str = "#======== File I/O ===============\n"
        fp.write(_str)
        video_path = os.path.join(raw_video_dir, raw_video_name + ".yuv")
        _str = "InputFile                     : " + video_path + "\n"
        fp.write(_str)
        _str = "InputBitDepth                 : 8           # Input bitdepth\n"
        fp.write(_str)
        _str = "InputChromaFormat             : 420         # Ratio of luminance to chrominance samples\n"
        fp.write(_str)
        _str = "FrameRate                     : 50          # Frame Rate per second\n"
        fp.write(_str)
        _str = "FrameSkip                     : 0           # Number of frames to be skipped in input\n"
        fp.write(_str)
        _str = "SourceWidth                   : " + w + "           # Input  frame width\n"
        fp.write(_str)
        _str = "SourceHeight                  : " + h + "           # Input  frame height\n"
        fp.write(_str)
        _str = "FramesToBeEncoded             : " + nfs + "         # Number of frames to be coded\n"
        fp.write(_str)
        _str = "Level                         : 3.1\n"
        fp.write(_str)

        fp.close()


def hevc_encode_with_hm(hm_enc_path, enc_config_path, vid_config_path,
                        src_yuv_path, dst_yuv_path, dst_bin_path,
                        log_txt_path=None):
    f = 64
    q = 32
    wdt, hgt = 480, 272
    if log_txt_path:
        command = f'{hm_enc_path} -c {enc_config_path} -c {vid_config_path} ' \
                  f'-i {src_yuv_path} -o {dst_yuv_path} -b {dst_bin_path} ' \
                  f'-f {f} -q {q} -wdt {wdt} -hgt {hgt} >{log_txt_path}'
    else:
        command = f'{hm_enc_path} -c {enc_config_path} -c {vid_config_path} ' \
                  f'-i {src_yuv_path} -o {dst_yuv_path} -b {dst_bin_path} ' \
                  f'-f {f} -q {q} -wdt {wdt} -hgt {hgt}'
    print(command)
    os.system(command)


def hevc_decode_with_hm(hm_dec_path, src_bin_path, dst_yuv_path, log_txt_path):
    command = f'{hm_dec_path} -b {src_bin_path} -o {dst_yuv_path} >{log_txt_path}'
    print(command)
    os.system(command)


if __name__ == '__main__':
    # src_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-PNG'
    # dst_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-MP4'
    #
    # mkdir(dst_root)
    #
    # src_frame_paths = sorted(glob.glob(osp.join(src_root, '*')))
    # for src_frame_path in src_frame_paths:
    #     dst_video_path = osp.join(dst_root, '{}.mp4'.format(os.path.basename(src_frame_path)))
    #     encode_video_with_ffmpeg('{}/%03d.png'.format(src_frame_path), dst_video_path, crf=0, fps=25, lib='libx264')

    # # HM encode
    # hm_enc_path = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/TAppEncoderStatic'
    # enc_config_path = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/enc_configs/encoder_lowdelay_P_main.cfg'
    # vid_config_path = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/vid_configs/seq_config.cfg'
    #
    # src_yuv_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-yuv-GT'
    # dst_yuv_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-yuv/QP32'
    # dst_bin_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-bin/QP32'
    # log_txt_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-txt/QP32'
    #
    # mkdir(dst_yuv_root)
    # mkdir(dst_bin_root)
    # mkdir(log_txt_root)
    #
    # src_yuv_paths = sorted(glob.glob(osp.join(src_yuv_root, '*.yuv')))
    # for src_yuv_path in src_yuv_paths:
    #     name = osp.basename(src_yuv_path)
    #     dst_yuv_path = osp.join(dst_yuv_root, name)
    #     dst_bin_path = osp.join(dst_bin_root, name.replace('yuv', 'bin'))
    #     log_txt_path = osp.join(log_txt_root, name.replace('yuv', 'txt'))
    #     hevc_encode_with_hm(hm_enc_path, enc_config_path, vid_config_path, src_yuv_path, dst_yuv_path, dst_bin_path, log_txt_path)

    # HM decode
    hm_dec_path = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/TAppDecoderStatic'

    src_bin_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-bin/QP32'
    dst_yuv_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-yuv/QP32'
    log_txt_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-txt/QP32'

    src_bin_paths = sorted(glob.glob(osp.join(src_bin_root, '*.bin')))
    for src_bin_path in src_bin_paths:
        name = osp.basename(src_bin_path)
        dst_yuv_path = osp.join(dst_yuv_root, name.replace('bin', 'yuv'))
        log_txt_path = osp.join(log_txt_root, name.replace('bin', 'txt'))
        hevc_decode_with_hm(hm_dec_path, src_bin_path, dst_yuv_path, log_txt_path)

    # raw_video_dir = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-yuv-GT'
    # cfg_save_dir  = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/vid_configs'
    # generate_video_cfgs(raw_video_dir, cfg_save_dir)

