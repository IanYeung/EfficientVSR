import yaml

from basicsr.data.cvcp_dataset import CVCPRecurrentDataset


if __name__ == '__main__':
    opt_str = r"""
name: Test
type: CVCPRecurrentDataset

dataroot_gt: /home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/1920x1088-LDR-gray
dataroot_lq: /home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-gray/QP32
dataroot_side: /home/xiyang/data1/datasets/Video-Super-Resolution-Datasets/BVI-DVC-HD/0480x0272-LDR-side/QP32

meta_info_file: /home/xiyang/Projects/BasicSR/basicsr/data/meta_info/meta_info_CVCP_GT.txt
filename_tmpl: '{}'
io_backend:
    type: disk

scale: 4

num_frame: 10
gt_size: 256
interval_list: [1]
random_reverse: false
use_flip: true
use_rot: true

phase: train
"""
    opt = yaml.safe_load(opt_str)
    datasets = CVCPRecurrentDataset(opt)
    data_dict = datasets.__getitem__(0)
    img_lqs = data_dict['lq']
    img_gts = data_dict['gt']

    img_masks = data_dict['mask']
    img_preds = data_dict['pred']
    img_resis = data_dict['resi']
    flows = data_dict['flow']
    key = data_dict['key']
