import os
import cv2
import glob
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    read_root = '/home/xiyang/Datasets/VSR-TEST/UDM10/GT'
    save_root = '/home/xiyang/Datasets/VSR-TEST/UDM10/Diff'
    seq_paths = sorted(glob.glob(os.path.join(read_root, '*')))
    for seq_path in seq_paths:
        mkdir(os.path.join(save_root, os.path.basename(seq_path)))
        img_paths = sorted(glob.glob(os.path.join(seq_path, '*.png')))
        for imfile1, imfile2 in zip(img_paths[:-1], img_paths[1:]):
            image1 = cv2.imread(imfile1)
            image2 = cv2.imread(imfile2)
            diff = np.abs(image1 - image2)
            save_path = os.path.join(save_root, os.path.basename(seq_path), os.path.basename(imfile1))
            cv2.imwrite(save_path, diff)
