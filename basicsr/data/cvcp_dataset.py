import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.data.transforms import augment_with_prior, paired_random_crop_with_prior
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY


def mv2mvs_ra(mvl0, mvl1):
    mvl0_ = mvl0.astype(np.float32)
    mvl0_ = mvl0_[np.newaxis, :, :, :]
    mvl0_[:, :, :, [0, 1]] = mvl0_[:, :, :, [1, 0]]

    mvl1_ = mvl1.astype(np.float32)
    mvl1_ = mvl1_[np.newaxis, :, :, :]
    mvl1_[:, :, :, [0, 1]] = mvl1_[:, :, :, [1, 0]]

    mvl0s_7 = np.zeros([7, mvl0_.shape[1], mvl0_.shape[2], 2]).astype(np.float32)
    # # frame 2
    pre_f_x = mvl0_[0, :, :, 0] / (mvl0_[0, :, :, 2] * -1.0)
    pre_f_y = mvl0_[0, :, :, 1] / (mvl0_[0, :, :, 2] * -1.0)
    MV0_u_m = np.where(mvl0_[0, :, :, 2] == -99)
    mvl0s_7[2, :, :, 0] = np.where(~np.isnan(pre_f_x), pre_f_x, 0)
    mvl0s_7[2, :, :, 1] = np.where(~np.isnan(pre_f_y), pre_f_y, 0)

    pre_f_x_MV1 = mvl1_[0, :, :, 0] / mvl1_[0, :, :, 2]
    pre_f_y_MV1 = mvl1_[0, :, :, 1] / mvl1_[0, :, :, 2]
    MV1_u_m = np.where(mvl1_[0, :, :, 2] == -99)
    mvl0s_7[4, :, :, 0] = np.where(~np.isnan(pre_f_x_MV1), pre_f_x_MV1, 0)
    mvl0s_7[4, :, :, 1] = np.where(~np.isnan(pre_f_y_MV1), pre_f_y_MV1, 0)

    ### complement
    mvl0s_7[2, :, :, 0][MV0_u_m] = mvl0s_7[4, :, :, 0][MV0_u_m] * -1
    mvl0s_7[2, :, :, 1][MV0_u_m] = mvl0s_7[4, :, :, 1][MV0_u_m] * -1

    mvl0s_7[4, :, :, 0][MV1_u_m] = mvl0s_7[2, :, :, 0][MV1_u_m] * -1
    mvl0s_7[4, :, :, 1][MV1_u_m] = mvl0s_7[2, :, :, 1][MV1_u_m] * -1

    ### scale to other frames
    mvl0s_7[1, :, :, :] = mvl0s_7[2, :, :, :] * 2.0
    mvl0s_7[0, :, :, :] = mvl0s_7[2, :, :, :] * 3.0

    mvl0s_7[5, :, :, :] = mvl0s_7[4, :, :, :] * 2.0
    mvl0s_7[6, :, :, :] = mvl0s_7[4, :, :, :] * 3.0

    mvl0s_7 = mvl0s_7 / (4.0 * 32.0)

    return torch.from_numpy(mvl0s_7).float()


def mv2mvs_ld(mv):
    mv_ = mv.astype(np.float32)
    mv_ = mv_[np.newaxis, :, :, :]
    mv_[:, :, :, [0, 1]] = mv_[:, :, :, [1, 0]]

    mvl0s_7 = np.zeros([7, mv_.shape[1], mv_.shape[2], 2]).astype(np.float32)
    # # frame 2
    pre_f_x = mv_[0, :, :, 0] / (mv_[0, :, :, 2] * -1.0)
    pre_f_y = mv_[0, :, :, 1] / (mv_[0, :, :, 2] * -1.0)

    mvl0s_7[2, :, :, 0] = np.where(~np.isnan(pre_f_x), pre_f_x, 0)
    mvl0s_7[2, :, :, 1] = np.where(~np.isnan(pre_f_y), pre_f_y, 0)

    mvl0s_7[1, :, :, :] = mvl0s_7[2, :, :, :] * 2.0
    mvl0s_7[0, :, :, :] = mvl0s_7[2, :, :, :] * 3.0

    mvl0s_7[4, :, :, :] = mvl0s_7[2, :, :, :] * -1.0
    mvl0s_7[5, :, :, :] = mvl0s_7[2, :, :, :] * -2.0
    mvl0s_7[6, :, :, :] = mvl0s_7[2, :, :, :] * -3.0

    mvl0s_7 = mvl0s_7 / (4.0 * 32.0)

    return torch.from_numpy(mvl0s_7).float()


def modify_mv_for_end_frames(i, mvs, max_idx):
    if i == 0:
        mvs[:, 0, :, :, :] = 0.0
        mvs[:, 1, :, :, :] = 0.0
        mvs[:, 2, :, :, :] = 0.0

    if i == 1:
        mvs[:, 0, :, :, :] = mvs[:, 2, :, :, :]
        mvs[:, 1, :, :, :] = mvs[:, 2, :, :, :]

    if i == 2:
        mvs[:, 0, :, :, :] = mvs[:, 1, :, :, :]

    if i == max_idx - 1:
        mvs[:, 4, :, :, :] = 0.0
        mvs[:, 5, :, :, :] = 0.0
        mvs[:, 6, :, :, :] = 0.0

    if i == max_idx - 2:
        mvs[:, 5, :, :, :] = mvs[:, 4, :, :, :]
        mvs[:, 6, :, :, :] = mvs[:, 4, :, :, :]

    if i == max_idx - 3:
        mvs[:, 6, :, :, :] = mvs[:, 5, :, :, :]

    return mvs


@DATASET_REGISTRY.register()
class CVCPDataset(data.Dataset):

    def __init__(self, opt):
        super(CVCPDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.flow_root = Path(opt['dataroot_flow']) if opt['dataroot_flow'] is not None else None
        assert opt['num_frame'] % 2 == 1, (f'num_frame should be odd number, but got {opt["num_frame"]}')
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        # each clip has 100 frames starting from 0 to 99
        while (start_frame_idx < 0) or (end_frame_idx > 99):
            center_frame_idx = random.randint(0, 99)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = center_frame_idx + self.num_half_frames * interval
        frame_name = f'{center_frame_idx:08d}'
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (f'Wrong length of neighbor list: {len(neighbor_list)}')

        # get the GT frame (as the center frame)
        if self.is_lmdb:
            img_gt_path = f'{clip_name}/{frame_name}'
        else:
            img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # get flows
        if self.flow_root is not None:
            img_flows = []
            # read previous flows
            for i in range(self.num_half_frames, 0, -1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_p{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_p{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)
            # read next flows
            for i in range(1, self.num_half_frames + 1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_n{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_n{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)

            # for random crop, here, img_flows and img_lqs have the same
            # spatial size
            img_lqs.extend(img_flows)

        # randomly crop
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path)
        if self.flow_root is not None:
            img_lqs, img_flows = img_lqs[:self.num_frame], img_lqs[self.num_frame:]

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        if self.flow_root is not None:
            img_results, img_flows = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'], img_flows)
        else:
            img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        if self.flow_root is not None:
            img_flows = img2tensor(img_flows)
            # add the zero center flow
            img_flows.insert(self.num_half_frames, torch.zeros_like(img_flows[0]))
            img_flows = torch.stack(img_flows, dim=0)

        # img_lqs: (t, c, h, w)
        # img_flows: (t, 2, h, w)
        # img_gt: (c, h, w)
        # key: str
        if self.flow_root is not None:
            return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key}
        else:
            return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class CVCPRecurrentDataset(data.Dataset):

    def __init__(self, opt):
        super(CVCPRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.side_root = Path(opt['dataroot_side']) if opt['dataroot_side'] is not None else None
        self.num_frame = opt['num_frame']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:05d}' for i in range(int(frame_num))])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 100 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 100 - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        if self.side_root:
            img_masks, img_preds, img_resis, flows = [], [], [], []

        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:05d}'
                img_gt_path = f'{clip_name}/{neighbor:05d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:05d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:05d}.png'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

            if self.side_root:
                # mask
                img_mask_path = self.side_root / clip_name / 'part_m' / f'{neighbor:05d}_M_mask.png'
                img_bytes = self.file_client.get(img_mask_path, 'mask')
                img_mask = imfrombytes(img_bytes, flag='grayscale', float32=True)
                img_mask = np.expand_dims(img_mask, axis=-1)
                img_masks.append(img_mask)

                # predicted image
                img_pred_path = self.side_root / clip_name / 'pred' / f'{neighbor:05d}_pred.png'
                img_bytes = self.file_client.get(img_pred_path, 'pred')
                img_pred = imfrombytes(img_bytes, flag='grayscale', float32=True)
                img_pred = np.expand_dims(img_pred, axis=-1)
                img_preds.append(img_pred)

                # residual
                img_resi_path = self.side_root / clip_name / 'res' / f'{neighbor:05d}_res.npy'
                img_resi = (np.load(img_resi_path)[:, :, 0]).astype(np.float32) / 255.
                img_resi = np.expand_dims(img_resi, axis=-1)
                img_resis.append(img_resi)

                # flow
                flow_path = self.side_root / clip_name / 'mvl0' / f'{neighbor:05d}_mvl0.npy'
                flow = (np.load(flow_path)[:, :, 0:2]).astype(np.float32)
                flows.append(flow)

        # randomly crop
        img_gts, img_lqs, img_masks, img_preds, img_resis, flows = paired_random_crop_with_prior(
            img_gts, img_lqs, img_masks, img_preds, img_resis, flows, gt_size, scale, img_gt_path
        )

        # augmentation - flip, rotate
        img_gts, img_lqs, img_masks, img_preds, img_resis, flows = augment_with_prior(
            img_gts, img_lqs, img_masks, img_preds, img_resis, flows,
            self.opt['use_flip'], self.opt['use_rot']
        )

        img_gts = torch.stack(img2tensor(img_gts), dim=0)
        img_lqs = torch.stack(img2tensor(img_lqs), dim=0)
        img_masks = torch.stack(img2tensor(img_masks, bgr2rgb=False), dim=0)
        img_preds = torch.stack(img2tensor(img_preds, bgr2rgb=False), dim=0)
        img_resis = torch.stack(img2tensor(img_resis, bgr2rgb=False), dim=0)
        flows = torch.stack(img2tensor(flows, bgr2rgb=False), dim=0)

        return {
            'lq': img_lqs,
            'gt': img_gts,
            'mask': img_masks,
            'pred': img_preds,
            'resi': img_resis,
            'flow': flows,
            'key': key
        }

    def __len__(self):
        return len(self.keys)
