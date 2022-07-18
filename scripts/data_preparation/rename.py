import os
import glob


def rename_inter4k(root):
    file_paths = sorted(glob.glob(os.path.join(root, '*.mp4')))
    for file_path in file_paths:
        clip_name = os.path.splitext(os.path.basename(file_path))[0]
        os.rename(file_path, os.path.join(root, '{}.mp4'.format(clip_name.zfill(4))))


def rename_bvi_dvc(root):
    file_paths = sorted(glob.glob(os.path.join(root, '*.mp4')))
    for idx, file_path in enumerate(file_paths):
        os.rename(file_path, os.path.join(root, '{:03d}.mp4'.format(idx)))


def rename_bvi_dvc_yuv(root):
    folder_paths = sorted(glob.glob(os.path.join(root, '*.yuv')))
    for folder_path in folder_paths:
        clip_name = os.path.splitext(os.path.basename(folder_path))[0]
        os.rename(folder_path, os.path.join(root, clip_name))


def rename_test_folders(root):
    folder_paths = sorted(glob.glob(os.path.join(root, '*', '*', '*')))
    for folder_path in folder_paths:
        frame_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))
        for idx, frame_path in enumerate(frame_paths):
            os.rename(frame_path, os.path.join(folder_path, '{:08d}.png'.format(idx)))


if __name__ == '__main__':

    # root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/1920x1088-HDR-MP4'
    # rename_bvi_dvc(root)

    # root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-540P/0960x0544-HDR-MP4'
    # rename_bvi_dvc(root)

    # root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-gray/QP32'
    # rename_bvi_dvc_yuv(root)

    root = '/home/xiyang/Datasets/VSR-TEST-Renamed'
    rename_test_folders(root)

