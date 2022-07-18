import torch
from torch import nn as nn
from torch.nn import functional as F

import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt

from basicsr.archs.raft_arch import RAFT, RAFT_SR, InputPadder, raft_inference
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.pwcnet_arch import PWCNet
from basicsr.archs.fastflownet_arch import FastFlowNet
from basicsr.archs.maskflownet_arch import MaskFlownet_S
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.utils.flow_util import flow_to_image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = True

    device = torch.device('cuda')
    img1_path = '../data/sintel/one.png'
    img2_path = '../data/sintel/two.png'

    img1 = img2tensor(cv2.imread(img1_path, cv2.IMREAD_UNCHANGED) / 255., bgr2rgb=True, float32=True).to(device)
    img2 = img2tensor(cv2.imread(img2_path, cv2.IMREAD_UNCHANGED) / 255., bgr2rgb=True, float32=True).to(device)

    load_path = '../../experiments/pretrained_models/flownet/maskflownet-ft-sintel.pth'
    maskflownet = MaskFlownet_S(load_path=load_path).to(device).eval()
    with torch.no_grad():
        flow0 = maskflownet(img1.unsqueeze(0), img2.unsqueeze(0))
    flow0 = flow0[0].permute(1, 2, 0).cpu().numpy()
    # plt.imshow(flow_to_image(flow0))
    # plt.title('MaskFlowNet')
    # plt.show()

    load_path = '../../experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth'
    spynet = SpyNet(load_path=load_path).to(device).eval()
    with torch.no_grad():
        flow1 = spynet(img1.unsqueeze(0), img2.unsqueeze(0))
    flow1 = flow1[0].permute(1, 2, 0).cpu().numpy()
    # plt.imshow(flow_to_image(flow1))
    # plt.title('SpyNet')
    # plt.show()

    pwcnet = PWCNet().to(device).eval()
    with torch.no_grad():
        flow2 = pwcnet(img1.unsqueeze(0), img2.unsqueeze(0))
    flow2 = flow2[0].permute(1, 2, 0).cpu().numpy()
    # plt.imshow(flow_to_image(flow2))
    # plt.title('PWCNet')
    # plt.show()

    load_path = '../../experiments/pretrained_models/flownet/fastflownet_ft_sintel.pth'
    fastflownet = FastFlowNet(load_path=load_path).to(device).eval()
    with torch.no_grad():
        flow3 = fastflownet(img1.unsqueeze(0), img2.unsqueeze(0))
    flow3 = flow3[0].permute(1, 2, 0).cpu().numpy()
    # plt.imshow(flow_to_image(flow3))
    # plt.title('FastFlowNet')
    # plt.show()

    # raftflownet = torch.nn.DataParallel(RAFT(model='small'))
    # raftflownet.load_state_dict(torch.load('../../experiments/pretrained_models/flownet/raft-small.pth'))
    # raftflownet = raftflownet.module
    # raftflownet = RAFT_SR(model='small', load_path='../../experiments/pretrained_models/flownet/raft-small.pth')
    # raftflownet = raftflownet.to(device).eval()
    # with torch.no_grad():
    #     # padder = InputPadder(img1[None].shape)
    #     # img1, img2 = padder.pad(img1[None], img2[None])
    #     flow4 = raftflownet(img1[None], img2[None], iters=10)
    #     # flow4 = raft_inference(raftflownet, img1[None], img2[None], iters=20, flow_init=None, upsample=True)
    # flow4 = flow4[0].permute(1, 2, 0).cpu().numpy()
    # plt.imshow(flow_to_image(flow4))
    # plt.title('RAFT')
    # plt.show()

    net = MaskFlownet_S()
    print('Params of MaskFlowNet: ', count_parameters(net))
    net = SpyNet()
    print('Params of SpyNet: ', count_parameters(net))
    net = PWCNet()
    print('Params of PWCNet: ', count_parameters(net))
    net = FastFlowNet()
    print('Params of FastFlowNet: ', count_parameters(net))
    # net = RAFT(model='small')
    # print('Params of RAFTFlowNet: ', count_parameters(net))

    b, c, h, w = 1, 3, 540, 960
    model_list = [
        SpyNet(),
        PWCNet(),
        FastFlowNet(),
        MaskFlownet_S(),
    ]
    for model in model_list:
        model.eval()
        model.to(device)

        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 20
        timings = np.zeros((repetitions, 1))
        # GPU-WARM-UP
        for _ in range(10):
            dummy_input1 = torch.randn(b, c, h, w).to(device)
            dummy_input2 = torch.randn(b, c, h, w).to(device)
            _ = model(dummy_input1, dummy_input2)

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                dummy_input1 = torch.randn(b, c, h, w).to(device)
                dummy_input2 = torch.randn(b, c, h, w).to(device)
                starter.record()
                _ = model(dummy_input1, dummy_input2)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(mean_syn)
