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
    BasicUniVSRFeatPropWithFlowDCN_Fast_FrmInp1, \
    BasicUniVSRFeatPropWithFlowDCN_Fast_FrmInp3
from basicsr.archs.basicvsr_arch import \
    BasicVSR, BasicVSRCouplePropV1, \
    BasicVSRCouplePropV2, BasicVSRCouplePropV3, \
    RealTimeBasicVSRCouplePropV1, RealTimeBasicVSRCouplePropV2, \
    RealTimeBasicVSRCouplePropV3, RealTimeBasicVSRCouplePropV4, \
    RealTimeBasicVSRCouplePropV2_FF, RealTimeBasicVSRCouplePropV3_FF
from basicsr.archs.basicvsrplusplus_arch import BasicVSRPlusPlus, BasicVSRPlusPlus3rdOrder
from basicsr.archs.efficient_vsr_arch import RLSP, RRN
from basicsr.archs.edvr_arch import EDVR
from basicsr.archs.ovsr_arch import OVSR
from basicsr.archs.dcn_vsr_arch import PCDAlignVSR, FlowDCNAlignVSR, FlowGuidedDeformAttnAlignVSR
from basicsr.archs.flow_vsr_arch import FlowFeaVSR, FlowFeaChannelTransformerVSR
from basicsr.archs.att_vsr_arch import PyramidAttentionAlignVSR
from basicsr.archs.ttvsr_arch import TTVSRNet
from basicsr.archs.vrt_arch import VRT


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    # device = torch.device('cuda:0')
    # b, n, c, h, w = 1, 5, 3, 2160 // 4, 3840 // 4
    # inp = torch.randn(b, n, c, h, w).to(device)
    #
    # model = BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=10).to(device)
    # flops = FlopCountAnalysis(model, inp)
    # print('Flops: {} G'.format(flops.total() / (n * 1e9)))

    # model = BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=20).to(device)
    # flops = FlopCountAnalysis(model, inp)
    # print('Flops: {} G'.format(flops.total() / (n * 1e9)))
    #
    # model = BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=30).to(device)
    # flops = FlopCountAnalysis(model, inp)
    # print('Flops: {} G'.format(flops.total() / (n * 1e9)))
    #
    # model = BasicUniVSRFeatPropWithSpyFlow_Fast(num_feat=32, num_block=10).to(device)
    # flops = FlopCountAnalysis(model, inp)
    # print('Flops: {} G'.format(flops.total() / (n * 1e9)))
    #
    # model = BasicUniVSRFeatPropWithFastFlow_Fast(num_feat=32, num_block=10).to(device)
    # flops = FlopCountAnalysis(model, inp)
    # print('Flops: {} G'.format(flops.total() / (n * 1e9)))
    #
    # model = BasicUniVSRFeatPropWithSpyFlowDCN_Fast(num_feat=32, num_block=10).to(device)
    # flops = FlopCountAnalysis(model, inp)
    # print('Flops: {} G'.format(flops.total() / (n * 1e9)))
    #
    # model = BasicUniVSRFeatPropWithFastFlowDCN_Fast(num_feat=32, num_block=10).to(device)
    # flops = FlopCountAnalysis(model, inp)
    # print('Flops: {} G'.format(flops.total() / (n * 1e9)))

    # model = BasicUniVSRFeatPropWithPCD(num_feat=64, num_block=20).to(device)
    # inp = torch.randn(1, 5, 3, 64, 64).to(device)
    # out = model(inp)
    # print(out.shape)
    # print(count_parameters(model))

    # model = BasicUniVSRFeatPropEarFusion(num_feat=64, num_block=20).to(device)
    # inp = torch.randn(1, 5, 3, 64, 64).to(device)
    # out = model(inp)
    # print(out.shape)
    # print(count_parameters(model))
    #
    # model = BasicUniVSRFeatPropSepSTSFusion(num_feat=64, num_block=20, depth_window_size=(1, 4, 4), num_heads=1).to(device)
    # inp = torch.randn(1, 5, 3, 64, 64).to(device)
    # out = model(inp)
    # print(out.shape)
    # print(count_parameters(model))
    #
    # model = BasicUniVSRFeatPropCoTSTAttFusion(num_feat=64, num_block=20, num_heads=1).to(device)
    # inp = torch.randn(1, 5, 3, 64, 64).to(device)
    # out = model(inp)
    # print(out.shape)
    # print(count_parameters(model))

    # model = BasicVSRPlusPlus3rdOrder(num_feat=64, num_blocks=7).to(device)
    # inp = torch.randn(1, 15, 3, 64, 64).to(device)
    # out = model(inp)
    # print(out.shape)
    # print(count_parameters(model))

    device = torch.device('cuda:0')
    model_list = [
        # BasicVSR(num_feat=64, num_block=8),
        # TTVSRNet(mid_channels=64, num_blocks=20, stride=4, keyframe_stride=3),
        # BasicVSRPlusPlus(num_feat=32, num_blocks=5),
        # RLSP(filters=32, state_dim=32, layers=20+2, kernel_size=3, factor=4),
        # RRN(n_c=32, n_b=10, scale=4),
        # RLSP(filters=128, state_dim=128, layers=20+2, kernel_size=3, factor=4),
        # RRN(n_c=128, n_b=10, scale=4),
        # OVSR(basic_filter=32, num_pb=2, num_sb=2, scale=4, num_frame=3, kind='local'),
        # BasicUniVSRFeatPropWithFlowDCN_Fast_FrmInp1(num_feat=32, num_block=10, one_stage_up=True),
        # BasicUniVSRFeatPropWithFlowDCN_Fast_FrmInp3(num_feat=32, num_block=10, one_stage_up=True),
        # VRT(upscale=4,
        #     img_size=[7, 64, 64],
        #     window_size=[6, 8, 8],
        #     depths=[4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2],
        #     indep_reconsts=[11, 12],
        #     embed_dims=[60, 60, 60, 60, 60, 60, 60, 90, 90, 90, 90, 90, 90],
        #     num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        #     spynet_path=None,
        #     pa_frames=2,
        #     deformable_groups=12
        #     ),
        # BasicVSRCouplePropV1(num_feat=64, num_block=8),
        # BasicVSRCouplePropV2(num_feat=64, num_block=8),
        # BasicVSRCouplePropV3(num_feat=64, num_block=8),
        # RealTimeBasicVSRCouplePropV1(num_feat=32, num_block=5),
        RealTimeBasicVSRCouplePropV2(num_feat=32, num_extract_block=0, num_block=2, one_stage_up=True),
        # RealTimeBasicVSRCouplePropV2_FF(num_feat=32, num_extract_block=0, num_block=5, one_stage_up=True),
        # RealTimeBasicVSRCouplePropV3(num_feat=32, num_extract_block=0, num_block=5, one_stage_up=True),
        # RealTimeBasicVSRCouplePropV3_FF(num_feat=32, num_extract_block=0, num_block=5, one_stage_up=True),
        # RealTimeBasicVSRCouplePropV4(num_feat=64, num_extract_block=0, num_block=8, num_points=4)
        # BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=10, one_stage_up=True),
        # BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=20, one_stage_up=True),
        # BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=30, one_stage_up=True),
        # BasicUniVSRFeatPropWithSpyFlow_Fast(num_feat=32, num_block=10, one_stage_up=True),
        BasicUniVSRFeatPropWithFastFlow_Fast(num_feat=32, num_block=10, one_stage_up=True),
        # BasicUniVSRFeatPropWithSpyFlowDCN_Fast(num_feat=32, num_block=10, one_stage_up=True),
        # BasicUniVSRFeatPropWithFastFlowDCN_Fast(num_feat=32, num_block=10, one_stage_up=True),
        # BasicUniVSRFeatPropDoubleFusion_Fast(num_feat=64, num_block=20),
        # BasicUniVSRFeatPropWithLocalCorr_Fast(num_feat=64, num_block=20, nbr_size=9),
        # BasicUniVSRFeatPropWithFlowDeformAtt_Fast_V1(num_feat=64, num_block=20, num_points=9),
        # BasicUniVSRFeatPropWithFlowDeformAtt_Fast_V2(num_feat=64, num_block=20, num_points=9)
    ]

    # b, n, c, h, w = 1, 20, 3, 720 // 4, 1280 // 4
    # b, n, c, h, w = 1, 20, 3, 1080 // 4, 1920 // 4
    b, n, c, h, w = 1, 10, 3, 2160 // 4, 3840 // 4

    for model in model_list:
        model.eval()
        model.to(device)

        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 20
        timings = np.zeros((repetitions, 1))

        # # GPU-WARM-UP
        # for _ in range(2):
        #     dummy_input = torch.randn(b, n, c, h, w).to(device)
        #     _ = model(dummy_input)
        #     del dummy_input
        #     torch.cuda.empty_cache()

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                dummy_input = torch.randn(b, n, c, h, w).to(device)
                starter.record()
                _ = model(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print('Inference time: {:.4f} ms.'.format(mean_syn / n))
        print('Inference fps: {:.4f}.'.format(1000 / (mean_syn / n)))
