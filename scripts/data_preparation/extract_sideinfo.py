import glob
import sys
import numpy as np
import os
import pickle
import cv2
import math
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def txt2npy(dir_path, file_name, res_path, frames, frame_height, frame_width):
    print(file_name + '...')
    # sys.exit(0)
    line_cnt = 0
    with open(dir_path + file_name) as fp:

        line = fp.readline()
        # cnt = 1

        QP_v = np.zeros([1, frames]).astype(np.int16)
        res = np.zeros([frame_height, frame_width, 1, frames]).astype(np.int16)
        PreF_mv = np.zeros([frame_height, frame_width, 3, frames]).astype(np.int16)  # 0-r-y 1-c-x
        AftF_mv = np.zeros([frame_height, frame_width, 3, frames]).astype(np.int16)  # 0-r-y 1-c-x
        unfiltered_fs = np.zeros([frame_height, frame_width, 1, frames]).astype(np.uint8)

        part_v = []
        for _ in range(frames):
            part_v.append([])

        lastMVINFO = False

        while line:
            # skip space line
            if len(line.strip()) > 0:
                # print(line.strip())
                pass
            else:
                line = fp.readline()
                continue

            # QP of every frame
            if line[:3] == 'POC':
                num_r = line.index('TId')
                idx_qp = int(line[3:num_r])
                num_l = line.index('QP ')
                num_r = line.index(' ) [DT')
                QP_v[:, idx_qp] = np.int16(line[num_l + 3:num_r])
                lastMVINFO = False

            # MVs Partition
            if line[:6] == 'MVINFO':
                line_b = line.split()
                # print(line_b)
                subX = int(line_b[4])
                subY = int(line_b[6])
                subW = int(line_b[8])
                subH = int(line_b[10])
                currPoc = int(line_b[14])
                Mv0_x = int(line_b[20])
                Mv0_y = int(line_b[22])
                Mv1_x = int(line_b[24])
                Mv1_y = int(line_b[26])

                L0_idx = int(line_b[16])
                L1_idx = int(line_b[18])

                if L0_idx == -1:
                    L0_idx = -99
                else:
                    L0_idx = L0_idx - currPoc

                if L1_idx == -1:
                    L1_idx = -99
                else:
                    L1_idx = L1_idx - currPoc

                # MV
                PreF_mv[subY:subY + subH, subX:subX + subW, 0, currPoc] = Mv0_y
                PreF_mv[subY:subY + subH, subX:subX + subW, 1, currPoc] = Mv0_x
                PreF_mv[subY:subY + subH, subX:subX + subW, 2, currPoc] = L0_idx

                AftF_mv[subY:subY + subH, subX:subX + subW, 0, currPoc] = Mv1_y
                AftF_mv[subY:subY + subH, subX:subX + subW, 1, currPoc] = Mv1_x
                AftF_mv[subY:subY + subH, subX:subX + subW, 2, currPoc] = L1_idx

                # part
                part_v[currPoc].append((subX, subY, subW, subH))

                # store left-top corner
                if not lastMVINFO:
                    left_top_p = (subX, subY, currPoc)
                lastMVINFO = True

            # Res
            if line[:8] == 'RESIINFO':
                line_b = line.split()
                w = int(line_b[2])
                tmp_vec = line_b[-int(w):]
                now_block_line_idx = int(line_b[6][:-1])

                try:
                    res[left_top_p[1] + now_block_line_idx, left_top_p[0]:left_top_p[0] + w, 0,
                    left_top_p[2]] = np.array(tmp_vec).astype(np.int16)
                except:
                    print(left_top_p)
                    print(np.array(tmp_vec).astype(np.int16))
                    print(res[left_top_p[1] + now_block_line_idx, left_top_p[0]:left_top_p[0] + w, 0,
                          left_top_p[2]].shape)
                    print(len(np.array(tmp_vec).astype(np.int16)))
                    print('w', w)
                    print(now_block_line_idx)
                    sys.exit(1)

                # res[left_top_p[1]+now_block_line_idx, left_top_p[0]:left_top_p[0]+w, 0, left_top_p[2]] = np.array(tmp_vec).astype(np.int16)

                # if w == now_block_line_idx + 1:
                #     del left_top_p
                # lastMVINFO = False

            if line[:8] == 'RECOINFO':
                line_b = line.split()
                w = int(line_b[2])
                tmp_vec = line_b[-int(w):]
                now_block_line_idx = int(line_b[6][:-1])

                unfiltered_fs[left_top_p[1] + now_block_line_idx, left_top_p[0]:left_top_p[0] + w, 0,
                left_top_p[2]] = np.array(tmp_vec).astype(np.uint8)

                if w == now_block_line_idx + 1:
                    del left_top_p
                lastMVINFO = False

            line = fp.readline()
            # print(line_cnt, '...', end="\r")
    fp.close()

    # create directory
    res_folder = res_path + file_name[:-4] + '/'
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    # save
    np.save(res_folder + file_name[:-17] + '_QP_v.npy', QP_v)

    resisual_path = res_folder + "res/"
    if not os.path.exists(resisual_path):
        os.mkdir(resisual_path)
    for i in range(frames):
        idx = "%05d" % i
        np.save(resisual_path + idx + '_res.npy', res[:, :, :, i])

    mvl0_path = res_folder + "mvl0/"
    if not os.path.exists(mvl0_path):
        os.mkdir(mvl0_path)
    for i in range(frames):
        idx = "%05d" % i
        np.save(mvl0_path + idx + '_mvl0.npy', PreF_mv[:, :, :, i])

    mvl1_path = res_folder + "mvl1/"
    if not os.path.exists(mvl1_path):
        os.mkdir(mvl1_path)
    for i in range(frames):
        idx = "%05d" % i
        np.save(mvl1_path + idx + '_mvl1.npy', AftF_mv[:, :, :, i])

    unfiltered_path = res_folder + "unfiltered/"
    if not os.path.exists(unfiltered_path):
        os.mkdir(unfiltered_path)
    for i in range(frames):
        idx = "%05d" % i
        cv2.imwrite(unfiltered_path + idx + '_unflt.png', unfiltered_fs[:, :, 0, i])

    pred_path = res_folder + "pred/"
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)
    for i in range(frames):
        idx = "%05d" % i
        tmp = unfiltered_fs[:, :, 0, i] - res[:, :, 0, i]
        cv2.imwrite(pred_path + idx + '_pred.png', tmp.astype(np.uint8))

    # pred_path = res_folder + "pred2/"
    # if not os.path.exists(pred_path):
    #     os.mkdir(pred_path)
    # for i in range(frames):
    #     idx = "%05d" % i
    #     tmp = unfiltered_fs[:,:,0,i] - res[:,:,0,i]
    #     mv_status = PreF_mv[:,:,2,i]
    #     tmp = np.where(mv_status==-99, unfiltered_fs[:,:,0,i], tmp)
    #     cv2.imwrite(pred_path + idx + '_pred2.png', tmp.astype(np.uint8))

    with open(res_folder + file_name[:-24] + '_part_v', 'wb') as fp:
        pickle.dump(part_v, fp)
    fp.close()


def readyuv420(filename, bitdepth, W, H, startframe, totalframe, show=False):
    # 从第startframe（含）开始读（0-based），共读totalframe帧

    uv_H = H // 2
    uv_W = W // 2

    if bitdepth == 8:
        Y = np.zeros((totalframe, H, W), np.uint8)
        U = np.zeros((totalframe, uv_H, uv_W), np.uint8)
        V = np.zeros((totalframe, uv_H, uv_W), np.uint8)
    elif bitdepth == 10:
        Y = np.zeros((totalframe, H, W), np.uint16)
        U = np.zeros((totalframe, uv_H, uv_W), np.uint16)
        V = np.zeros((totalframe, uv_H, uv_W), np.uint16)

    bytes2num = partial(int.from_bytes, byteorder='little', signed=False)

    bytesPerPixel = math.ceil(bitdepth / 8)
    seekPixels = startframe * H * W * 3 // 2
    fp = open(filename, 'rb')
    fp.seek(bytesPerPixel * seekPixels)

    for i in range(totalframe):

        for m in range(H):
            for n in range(W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    Y[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    Y[i, m, n] = np.uint16(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    U[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    U[i, m, n] = np.uint16(pel)

        for m in range(uv_H):
            for n in range(uv_W):
                if bitdepth == 8:
                    pel = bytes2num(fp.read(1))
                    V[i, m, n] = np.uint8(pel)
                elif bitdepth == 10:
                    pel = bytes2num(fp.read(2))
                    V[i, m, n] = np.uint16(pel)

        if show:
            pass

    if totalframe == 1:
        return Y[0], U[0], V[0]
    else:
        return Y, U, V


def generate_partition_mask(quadruples_list, ori_y, h, w):
    boundary_mask = np.zeros([h, w]).astype(np.uint8)
    mean_mask = np.zeros([h, w]).astype(np.uint8)
    for quad in quadruples_list:
        subX = quad[0]
        subY = quad[1]
        subW = quad[2]
        subH = quad[3]

        endX = subX + subW
        endY = subY + subH
        if endX >= w:
            endX = w - 1
        if endY >= h:
            endY = h - 1
        # draw rectangle in boundary mask
        boundary_mask[subY, subX:endX] = 255
        boundary_mask[endY, subX:endX] = 255
        boundary_mask[subY:endY, subX] = 255
        boundary_mask[subY:endY, endX] = 255

        # cal mean
        mean_mask[subY:endY, subX:endX] = np.mean(ori_y[subY:endY, subX:endX])
    return mean_mask, boundary_mask


def extract(filename, SIDEpath_, YUVpath_, LRpath, frames, frame_height, frame_width):
    print(filename + '...')
    # frames = 32
    y, _, _ = readyuv420(YUVpath_ + filename, 8, frame_width, frame_height, 0, frames, False)

    tmp_interim_name = filename[:-4]
    real_side_path = SIDEpath_ + tmp_interim_name + '/'

    with open(real_side_path + tmp_interim_name[0:-20] + '_part_v', 'rb') as part_fp:
        partition_list = pickle.load(part_fp)
    part_fp.close()

    # make result_root_folder
    res_folder = LRpath + filename + '/'
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    # make side_info_folder
    info_folder = SIDEpath_ + tmp_interim_name + '/part_m/'
    if not os.path.exists(info_folder):
        os.mkdir(info_folder)

    for i in range(frames):
        s = "%05d" % i

        tmp_im = Image.fromarray(y[i, :, :])
        tmp_im.save(res_folder + '/' + s + '.png')

        # 2 partition masks (mean and boundary)
        if frame_height == 272:
            one_frame_y = y[i, :, :]
            one_frame_y[-1, :] = one_frame_y[-3, :]
            one_frame_y[-2, :] = one_frame_y[-3, :]
        elif frame_height == 184:
            one_frame_y = y[i, :, :]
            one_frame_y[-1, :] = one_frame_y[-5, :]
            one_frame_y[-2, :] = one_frame_y[-5, :]
            one_frame_y[-3, :] = one_frame_y[-5, :]
            one_frame_y[-4, :] = one_frame_y[-5, :]
        else:
            one_frame_y = y[i, :, :]
        M_mask, B_mask = generate_partition_mask(partition_list[i], one_frame_y, frame_height, frame_width)
        tmp_im = Image.fromarray(B_mask)
        tmp_im.save(info_folder + s + '_B_mask.png')
        tmp_im = Image.fromarray(M_mask)
        tmp_im.save(info_folder + s + '_M_mask.png')


def main():

    txt2npy('/home/xiyang/data0/datasets/Video-Compression-Datasets/MFQEv2/test_18/qp37-ra/',
            'BasketballPass_416x240_500.txt',
            '/home/xiyang/data0/datasets/Video-Compression-Datasets/MFQEv2/test_18/qp37-ra/',
            frames=500,
            frame_height=240,
            frame_width=416)

    extract('BasketballPass_416x240_500.yuv',
            '/home/xiyang/data0/datasets/Video-Compression-Datasets/MFQEv2/test_18/qp37-ra/',
            '/home/xiyang/data0/datasets/Video-Compression-Datasets/MFQEv2/test_18/qp37-ra/',
            '/home/xiyang/data0/datasets/Video-Compression-Datasets/MFQEv2/test_18/qp37-ra/lr_grey/',
            frames=500,
            frame_height=240,
            frame_width=416)


if __name__ == '__main__':
    # main()

    txt_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-txt/QP32/'
    yuv_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-yuv/QP32/'
    side_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-side/QP32/'
    gray_root = '/home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-gray/QP32/'

    mkdir(side_root)
    mkdir(gray_root)

    log_txt_paths = sorted(glob.glob(os.path.join(txt_root, '*.txt')))
    for log_txt_path in log_txt_paths:
        txt_file_name = os.path.basename(log_txt_path)
        yuv_file_name = txt_file_name.replace('.txt', '.yuv')
        txt2npy(txt_root, txt_file_name, side_root, frames=64, frame_width=480, frame_height=272)
        extract(yuv_file_name, side_root, yuv_root, gray_root, frames=64, frame_width=480, frame_height=272)
