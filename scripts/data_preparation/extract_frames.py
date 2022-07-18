import os
import os.path as osp
import sys
import glob
import numpy as np
import cv2
import imageio
import ffmpeg
import matplotlib.pyplot as plt


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def hdr_to_ldr(src_video_path, dst_video_path):
    command = f'ffmpeg -i {src_video_path} -c:v libx264 -crf 0 -vf format=yuv420p {dst_video_path}'
    print(command)
    os.system(command)


def mp4_to_yuv(src_video_path, dst_video_path):
    command = f'ffmpeg -i {src_video_path} {dst_video_path}'
    print(command)
    os.system(command)


def extract_frames_from_single_video_yuv(video_path, out_path, mode='png'):

    assert mode == 'npy' or mode == 'npz' or mode == 'png'
    mkdir(out_path)

    video = imageio.get_reader(video_path, format='ffmpeg', mode='I', dtype=np.uint8)
    fps = video.get_meta_data()['fps']
    w, h = video.get_meta_data()['size']
    video.close()
    print('fps: {}'.format(fps))
    print('h: {}'.format(h))
    print('w: {}'.format(w))

    process = (
        ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
            .run_async(pipe_stdout=True)
    )

    index = 0

    while True:
        in_bytes_Y = process.stdout.read(w * h)
        in_bytes_U = process.stdout.read(w // 2 * h // 2)
        in_bytes_V = process.stdout.read(w // 2 * h // 2)
        if not in_bytes_Y:
            break
        image_Y = (
            np
                .frombuffer(in_bytes_Y, np.uint8)
                .reshape([h, w])
        )
        image_U = (
            np
                .frombuffer(in_bytes_U, np.uint8)
                .reshape([h // 2, w // 2])
        )
        image_V = (
            np
                .frombuffer(in_bytes_V, np.uint8)
                .reshape([h // 2, w // 2])
        )
        UMatrix = np.zeros([h // 2, w], dtype=np.uint8)
        VMatrix = np.zeros([h // 2, w], dtype=np.uint8)
        UMatrix[:, 0::2] = image_U
        UMatrix[:, 1::2] = image_U
        VMatrix[:, 0::2] = image_V
        VMatrix[:, 1::2] = image_V
        YUV = np.zeros([h, w, 3], dtype=np.uint8)
        YUV[:, :, 0] = image_Y
        YUV[0::2, :, 1] = UMatrix
        YUV[1::2, :, 1] = UMatrix
        YUV[0::2, :, 2] = VMatrix
        YUV[1::2, :, 2] = VMatrix

        index += 1
        if mode == 'png':
            cv2.imwrite(osp.join(out_path, '{:03d}.png'.format(index)), YUV)
        elif mode == 'npy':
            np.save(osp.join(out_path, '{:03d}.npy'.format(index)), YUV)
        elif mode == 'npz':
            np.savez_compressed(osp.join(out_path, '{:03d}'.format(index)), img=YUV)


def extract_frames_from_single_video_rgb(video_path, out_path):
    mkdir(out_path)
    video = imageio.get_reader(video_path, format='ffmpeg', mode='I', dtype=np.uint8)
    fps = video.get_meta_data()['fps']
    w, h = video.get_meta_data()['size']
    video.close()
    print('fps: {}'.format(fps))
    print('h: {}'.format(h))
    print('w: {}'.format(w))

    process = (
        ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True)
    )

    k = 0
    while True:
        in_bytes = process.stdout.read(w * h * 3)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([h, w, 3])
        cv2.imwrite(os.path.join(out_path, '{:08d}.png'.format(k)), frame)
        k += 1

    process.wait()
    print('total {} frames'.format(k - 1))


def extract_frames_from_single_video_y_only(video_path, out_path, mode='png'):
    mkdir(out_path)
    video = imageio.get_reader(video_path, format='ffmpeg', mode='I', dtype=np.uint8)
    fps = video.get_meta_data()['fps']
    w, h = video.get_meta_data()['size']
    video.close()
    print('fps: {}'.format(fps))
    print('h: {}'.format(h))
    print('w: {}'.format(w))

    process = (
        ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='yuv420p')
            .run_async(pipe_stdout=True)
    )

    index = 0

    while True:
        in_bytes_Y = process.stdout.read(w * h)
        in_bytes_U = process.stdout.read(w // 2 * h // 2)
        in_bytes_V = process.stdout.read(w // 2 * h // 2)
        if not in_bytes_Y:
            break
        image_Y = (
            np
                .frombuffer(in_bytes_Y, np.uint8)
                .reshape([h, w])
        )

        cv2.imwrite(osp.join(out_path, '{:05d}.png'.format(index)), image_Y)
        index += 1


def extract_first_frame_from_video_rgb(video_path, out_path):
    mkdir(out_path)
    video = imageio.get_reader(video_path, format='ffmpeg', mode='I', dtype=np.uint8)
    fps = video.get_meta_data()['fps']
    w, h = video.get_meta_data()['size']
    video.close()
    print('fps: {}'.format(fps))
    print('h: {}'.format(h))
    print('w: {}'.format(w))

    process = (
        ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True)
    )

    video_name = os.path.basename(video_path).split('.')[0]
    for i in range(1):
        in_bytes = process.stdout.read(w * h * 3)
        if not in_bytes:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape([h, w, 3])
        cv2.imwrite(os.path.join(out_path, '{}.png'.format(video_name)), frame)

    process.stdout.close()
    process.wait()


if __name__ == '__main__':

    # # HDR to LDR
    # src_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/1920x1088-HDR-MP4'
    # dst_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/1920x1088-LDR-MP4'
    #
    # mkdir(dst_root)
    #
    # src_video_paths = sorted(glob.glob(osp.join(src_root, '*.mp4')))
    # for src_video_path in src_video_paths:
    #     dst_video_path = osp.join(dst_root, osp.basename(src_video_path))
    #     hdr_to_ldr(src_video_path, dst_video_path)

    # # extract frames
    # src_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-540P/0960x0544-LDR-MP4'
    # dst_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-540P/0960x0544-LDR-PNG'
    #
    # mkdir(dst_root)
    # src_video_paths = sorted(glob.glob(osp.join(src_root, '*.mp4')))
    # for src_video_path in src_video_paths:
    #     dst_frame_path = osp.join(dst_root, osp.basename(src_video_path).split('.')[0])
    #     extract_frames_from_single_video_rgb(src_video_path, dst_frame_path)

    # # extract frames
    # src_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/1920x1088-LDR-MP4'
    # dst_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/1920x1088-LDR-gray'
    #
    # mkdir(dst_root)
    # src_video_paths = sorted(glob.glob(osp.join(src_root, '*.mp4')))
    # for src_video_path in src_video_paths:
    #     dst_frame_path = osp.join(dst_root, osp.basename(src_video_path).split('.')[0])
    #     extract_frames_from_single_video_y_only(src_video_path, dst_frame_path)

    # bicubic down-sample
    # matlab_scripts/generate_LR_BVI_DVC.m

    # # extract frames
    # src_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-MP4'
    # dst_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-yuv'
    #
    # mkdir(dst_root)
    # src_video_paths = sorted(glob.glob(osp.join(src_root, '*.mp4')))
    # for src_video_path in src_video_paths:
    #     dst_frame_path = osp.join(dst_root, osp.basename(src_video_path).replace('mp4', 'yuv'))
    #     mp4_to_yuv(src_video_path, dst_frame_path)

    # # extract frames
    # src_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/Inter4K/Inter4K/60fps/UHD'
    # dst_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/Inter4K-PNG/UHD'
    #
    # mkdir(dst_root)
    # src_video_paths = sorted(glob.glob(osp.join(src_root, '*.mp4')))
    # for src_video_path in src_video_paths:
    #     extract_first_frame_from_video_rgb(src_video_path, dst_root)

    # # extract frames
    # seq_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/Inter4K-Thumbnail-Test'
    # include_list = sorted(glob.glob(osp.join(seq_root, '*')))
    # include_list = [osp.basename(seq).split('.')[0] for seq in include_list]
    #
    # src_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/Inter4K/Inter4K/60fps/UHD'
    # dst_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/Inter4K-PNG/UHD'
    #
    # mkdir(dst_root)
    # src_video_paths = sorted(glob.glob(osp.join(src_root, '*.mp4')))
    #
    # for src_video_path in src_video_paths:
    #     video_name = osp.basename(src_video_path).split('.')[0]
    #     if video_name not in include_list:
    #         continue
    #     dst_frame_path = osp.join(dst_root, video_name)
    #     extract_frames_from_single_video_rgb(src_video_path, dst_frame_path)

    from shutil import copy

    mode = 'Gaussian4xLR'
    src_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/Inter4K-PNG/UHD/{}'.format(mode)
    dst_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/Inter4K-PNG/UHD-100F/{}'.format(mode)

    mkdir(dst_root)
    seq_list = sorted(glob.glob(osp.join(src_root, '*')))
    for src_seq_path in seq_list:
        seq_name = osp.basename(src_seq_path)
        print(seq_name)
        dst_seq_path = osp.join(dst_root, seq_name)
        mkdir(dst_seq_path)
        frame_path = sorted(glob.glob(osp.join(src_seq_path, '*.png')))
        for i in range(100):
            copy(src=frame_path[i], dst=dst_seq_path)




