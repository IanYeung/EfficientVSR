import torch
from torch import nn as nn
from torch.nn import functional as F

import numpy as np

from fvcore.nn import FlopCountAnalysis
from pthflops import count_ops
from thop import profile

from basicsr.archs.online_vsr_arch import \
    BasicUniVSRFeatPropWithoutAlign_Fast, \
    BasicUniVSRFeatPropWithSpyFlow_Fast, \
    BasicUniVSRFeatPropWithPWCFlow_Fast, \
    BasicUniVSRFeatPropWithFastFlow_Fast, \
    BasicUniVSRFeatPropWithSpyFlowDCN_Fast, \
    BasicUniVSRFeatPropWithFastFlowDCN_Fast, \
    BasicUniVSRFeatPropWithFlowDeformAtt_Fast_V1, \
    BasicUniVSRFeatPropWithFlowDeformAtt_Fast_V2, \
    BasicUniVSRFeatPropWithLocalCorr_Fast
from basicsr.archs.basicvsr_arch import \
    BasicVSR, BasicVSRCouplePropV1, \
    BasicVSRCouplePropV2, BasicVSRCouplePropV3, \
    RealTimeBasicVSRCouplePropV1, RealTimeBasicVSRCouplePropV2, \
    RealTimeBasicVSRCouplePropV3, RealTimeBasicVSRCouplePropV4
from basicsr.archs.basicvsrplusplus_arch import BasicVSRPlusPlus, BasicVSRPlusPlus3rdOrder
from basicsr.archs.efficient_vsr_arch import RLSP, RRN
from basicsr.archs.edvr_arch import EDVR
from basicsr.archs.dcn_vsr_arch import PCDAlignVSR, FlowDCNAlignVSR, FlowGuidedDeformAttnAlignVSR
from basicsr.archs.flow_vsr_arch import FlowFeaVSR, FlowFeaChannelTransformerVSR
from basicsr.archs.att_vsr_arch import PyramidAttentionAlignVSR
from basicsr.archs.ttvsr_arch import TTVSRNet
from basicsr.archs.vrt_arch import VRT


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    device = torch.device('cuda:0')
    b, n, c, h, w = 1, 5, 3, 2160 // 4, 3840 // 4
    inp = torch.randn(b, n, c, h, w).to(device)

    model = BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=10).to(device)
    flops = FlopCountAnalysis(model, inp)
    print('Flops: {} G'.format(flops.total() / (n * 1e9)))

    model = BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=20).to(device)
    flops = FlopCountAnalysis(model, inp)
    print('Flops: {} G'.format(flops.total() / (n * 1e9)))

    model = BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=30).to(device)
    flops = FlopCountAnalysis(model, inp)
    print('Flops: {} G'.format(flops.total() / (n * 1e9)))

    model = BasicUniVSRFeatPropWithSpyFlow_Fast(num_feat=32, num_block=10).to(device)
    flops = FlopCountAnalysis(model, inp)
    print('Flops: {} G'.format(flops.total() / (n * 1e9)))

    model = BasicUniVSRFeatPropWithFastFlow_Fast(num_feat=32, num_block=10).to(device)
    flops = FlopCountAnalysis(model, inp)
    print('Flops: {} G'.format(flops.total() / (n * 1e9)))

    model = BasicUniVSRFeatPropWithSpyFlowDCN_Fast(num_feat=32, num_block=10).to(device)
    flops = FlopCountAnalysis(model, inp)
    print('Flops: {} G'.format(flops.total() / (n * 1e9)))

    model = BasicUniVSRFeatPropWithFastFlowDCN_Fast(num_feat=32, num_block=10).to(device)
    flops = FlopCountAnalysis(model, inp)
    print('Flops: {} G'.format(flops.total() / (n * 1e9)))

