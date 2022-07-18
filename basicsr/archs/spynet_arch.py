import math
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.ops.correlation import correlation
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import flow_warp


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


@ARCH_REGISTRY.register()
class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2, int(math.floor(ref[0].size(2) / 2.0)), int(math.floor(ref[0].size(3) / 2.0))]
        )

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow = self.process(ref, supp)
        flow = F.interpolate(input=flow, size=(h, w), mode='bilinear', align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow


# backwarp_tenGrid = {}
#
#
# def backwarp(tenInput, tenFlow):
#     if str(tenFlow.shape) not in backwarp_tenGrid:
#         tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).\
#             view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
#         tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).\
#             view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
#
#         backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
#
#     tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
#                          tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
#
#     return F.grid_sample(input=tenInput,
#                          grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
#                          mode='bilinear', padding_mode='border', align_corners=False)
#
#
# class SpyNet(torch.nn.Module):
#
#     def __init__(self, load_path=None):
#         super().__init__()
#
#         class Preprocess(torch.nn.Module):
#
#             def __init__(self):
#                 super().__init__()
#
#             def forward(self, tenInput):
#                 tenInput = tenInput.flip([1])
#                 tenInput = tenInput - torch.tensor(data=[0.485, 0.456, 0.406], dtype=tenInput.dtype,
#                                                    device=tenInput.device).view(1, 3, 1, 1)
#                 tenInput = tenInput * torch.tensor(data=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=tenInput.dtype,
#                                                    device=tenInput.device).view(1, 3, 1, 1)
#
#                 return tenInput
#
#         class Basic(torch.nn.Module):
#             def __init__(self, intLevel):
#                 super().__init__()
#
#                 self.netBasic = torch.nn.Sequential(
#                     torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
#                     torch.nn.ReLU(inplace=False),
#                     torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
#                     torch.nn.ReLU(inplace=False),
#                     torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
#                     torch.nn.ReLU(inplace=False),
#                     torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
#                     torch.nn.ReLU(inplace=False),
#                     torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
#                 )
#
#             def forward(self, tenInput):
#                 return self.netBasic(tenInput)
#
#         self.netPreprocess = Preprocess()
#
#         self.netBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(6)])
#
#         self.load_state_dict({
#             strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
#             torch.hub.load_state_dict_from_url(
#                 url='http://content.sniklaus.com/github/pytorch-spynet/network-sintel-final.pytorch',
#                 file_name='spynet-sintel-final').items()
#             }
#         )
#
#     def process(self, tenOne, tenTwo):
#         tenFlow = []
#
#         tenOne = [self.netPreprocess(tenOne)]
#         tenTwo = [self.netPreprocess(tenTwo)]
#
#         for intLevel in range(5):
#             if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
#                 tenOne.insert(0, F.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2, count_include_pad=False))
#                 tenTwo.insert(0, F.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2, count_include_pad=False))
#
#         tenFlow = tenOne[0].new_zeros([tenOne[0].shape[0], 2, int(math.floor(tenOne[0].shape[2] / 2.0)),
#                                        int(math.floor(tenOne[0].shape[3] / 2.0))])
#
#         for intLevel in range(len(tenOne)):
#             tenUpsampled = F.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
#
#             if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]:
#                 tenUpsampled = F.pad(input=tenUpsampled, pad=[0, 0, 0, 1], mode='replicate')
#             if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]:
#                 tenUpsampled = F.pad(input=tenUpsampled, pad=[0, 1, 0, 0], mode='replicate')
#
#             tenFlow = self.netBasic[intLevel](
#                 torch.cat(
#                     [tenOne[intLevel], backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled), tenUpsampled], dim=1
#                 )
#             ) + tenUpsampled
#
#         return tenFlow
#
#     def forward(self, tenOne, tenTwo):
#         assert (tenOne.shape[1] == tenTwo.shape[1])
#         assert (tenOne.shape[2] == tenTwo.shape[2])
#
#         intHeight = tenOne.shape[2]
#         intWidth = tenOne.shape[3]
#
#         tenPreprocessedOne = tenOne
#         tenPreprocessedTwo = tenTwo
#
#         intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
#         intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))
#
#         tenPreprocessedOne = F.interpolate(input=tenPreprocessedOne,
#                                            size=(intPreprocessedHeight, intPreprocessedWidth),
#                                            mode='bilinear', align_corners=False)
#         tenPreprocessedTwo = F.interpolate(input=tenPreprocessedTwo,
#                                            size=(intPreprocessedHeight, intPreprocessedWidth),
#                                            mode='bilinear', align_corners=False)
#
#         tenFlow = F.interpolate(input=self.process(tenPreprocessedOne, tenPreprocessedTwo),
#                                 size=(intHeight, intWidth),
#                                 mode='bilinear',
#                                 align_corners=False)
#
#         tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
#         tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
#
#         return tenFlow
