import glob
from os import path as osp
from PIL import Image

from basicsr.utils import scandir


def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = 'datasets/DIV2K/DIV2K_train_HR_sub/'
    meta_info_txt = 'basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt'

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


def generate_meta_info_bvidvc():
    """Generate meta info for CVCP dataset.
    """

    gt_folder = '/home/xiyang/Datasets/BVI-DVC-540P/GT'
    meta_info_txt = '/home/xiyang/Projects/BasicSR/basicsr/data/meta_info/meta_info_BVIDVC_GT.txt'

    seq_list = sorted(glob.glob(osp.join(gt_folder, '*')))

    with open(meta_info_txt, 'w') as f:
        for idx, seq_path in enumerate(seq_list):
            num_frame = len(glob.glob(osp.join(seq_path, '*.png')))
            if num_frame <= 0:
                continue
            img = Image.open(osp.join(seq_path, '00000000.png'))
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{osp.basename(seq_path)} {num_frame:03d} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


def generate_meta_info_reds():
    """Generate meta info for REDS dataset.
    """

    gt_folder = '/home/xiyang/Datasets/REDS/train_sharp_sub'
    meta_info_txt = '/home/xiyang/Projects/BasicSR/basicsr/data/meta_info/meta_info_REDS_GT_sub.txt'

    seq_list = sorted(glob.glob(osp.join(gt_folder, '*')))

    with open(meta_info_txt, 'w') as f:
        for idx, seq_path in enumerate(seq_list):
            num_frame = len(glob.glob(osp.join(seq_path, '*.png')))
            if num_frame <= 0:
                continue
            img = Image.open(osp.join(seq_path, '00000000.png'))
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{osp.basename(seq_path)} {num_frame:03d} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


if __name__ == '__main__':
    # generate_meta_info_div2k()
    # generate_meta_info_bvidvc()
    generate_meta_info_reds()
