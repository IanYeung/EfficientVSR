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
    BasicUniVSRFeatPropWithPCDAlign, \
    BasicUniVSRFeatPropWithFlowDCN, \
    BasicUniVSRFeatPropWithSpyFlow, \
    BasicUniVSRFeatPropWithSpyFlow_FGAC
from basicsr.archs.basicvsr_arch import \
    BasicVSR, IconVSR, \
    BasicVSR_DirectUp, IconVSR_DirectUp, \
    BasicVSR_DirectUp_NoFlow, IconVSR_DirectUp_NoFlow, \
    BasicVSRCouplePropV1, BasicVSRCouplePropV2, BasicVSRCouplePropV3, \
    RealTimeBasicVSRCouplePropV1, RealTimeBasicVSRCouplePropV2, \
    RealTimeBasicVSRCouplePropV3, RealTimeBasicVSRCouplePropV4, \
    RealTimeBasicVSRCoupleProp_NoFlow
from basicsr.archs.basicvsrplusplus_arch import BasicVSRPlusPlus, BasicVSRPlusPlus_DirectUp
from basicsr.archs.efficient_vsr_arch import RLSP, RRN
from basicsr.archs.edvr_arch import EDVR
from basicsr.archs.dcn_vsr_arch import PCDAlignVSR, FlowDCNAlignVSR, FlowGuidedDeformAttnAlignVSR
from basicsr.archs.flow_vsr_arch import FlowFeaVSR, FlowFeaChannelTransformerVSR
from basicsr.archs.att_vsr_arch import PyramidAttentionAlignVSR
from basicsr.archs.ttvsr_arch import TTVSRNet
from basicsr.archs.vrt_arch import VRT


if __name__ == '__main__':

    device = torch.device('cuda:0')
    model_list = [
        # RLSP(filters=32, state_dim=32, layers=20+2, kernel_size=3, factor=4),
        # RRN(n_c=32, n_b=10, scale=4),
        # RLSP(filters=128, state_dim=128, layers=20+2, kernel_size=3, factor=4),
        # RRN(n_c=128, n_b=10, scale=4),
        # BasicVSR(num_feat=32, num_block=10),
        # BasicVSR_DirectUp(num_feat=32, num_block=10),
        # BasicVSR_DirectUp_NoFlow(num_feat=32, num_block=10),
        # IconVSR(num_feat=32, num_block=10),
        # IconVSR_DirectUp(num_feat=32, num_block=10),
        # IconVSR_DirectUp_NoFlow(num_feat=32, num_block=10),
        # RealTimeBasicVSRCoupleProp_NoFlow(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True),
        # RealTimeBasicVSRCoupleProp_NoFlow(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=False),
        # RealTimeBasicVSRCouplePropV2(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=True),
        # RealTimeBasicVSRCouplePropV2(num_feat=32, num_extract_block=0, num_block=10, one_stage_up=False),
        # RealTimeBasicVSRCouplePropV3(num_feat=32, num_extract_block=0, num_block=10, deformable_groups=8, one_stage_up=True),
        # RealTimeBasicVSRCouplePropV3(num_feat=32, num_extract_block=0, num_block=10, deformable_groups=8, one_stage_up=False),
        BasicUniVSRFeatPropWithPCDAlign(num_feat=32, num_block=10, one_stage_up=True),
        BasicUniVSRFeatPropWithSpyFlow_FGAC(num_feat=32, num_block=10, one_stage_up=True),
        BasicUniVSRFeatPropWithSpyFlowDCN_Fast(num_feat=32, num_block=10, one_stage_up=True),
        BasicUniVSRFeatPropWithFastFlowDCN_Fast(num_feat=32, num_block=10, one_stage_up=True),
        # TTVSRNet(mid_channels=64, num_blocks=20, stride=4, keyframe_stride=3),
        # BasicVSRPlusPlus(num_feat=32, num_blocks=2),
        # BasicVSRPlusPlus_DirectUp(num_feat=32, num_blocks=2),
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
        # RealTimeBasicVSRCouplePropV2(num_feat=64, num_extract_block=0, num_block=8),
        # RealTimeBasicVSRCouplePropV3(num_feat=64, num_extract_block=0, num_block=8),
        # RealTimeBasicVSRCouplePropV4(num_feat=64, num_extract_block=0, num_block=8, num_points=4)
        # BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=10, one_stage_up=True),
        # BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=20, one_stage_up=True),
        # BasicUniVSRFeatPropWithoutAlign_Fast(num_feat=32, num_block=30, one_stage_up=True),
        # BasicUniVSRFeatPropWithSpyFlow_Fast(num_feat=32, num_block=10, one_stage_up=True),
        # BasicUniVSRFeatPropWithFastFlow_Fast(num_feat=32, num_block=10, one_stage_up=True),
        # BasicUniVSRFeatPropWithSpyFlowDCN_Fast(num_feat=32, num_block=10, one_stage_up=True),
        # BasicUniVSRFeatPropWithFastFlowDCN_Fast(num_feat=32, num_block=10, one_stage_up=True),
        # BasicUniVSRFeatPropDoubleFusion_Fast(num_feat=64, num_block=20),
        # BasicUniVSRFeatPropWithLocalCorr_Fast(num_feat=64, num_block=20, nbr_size=9),
        # BasicUniVSRFeatPropWithFlowDeformAtt_Fast_V1(num_feat=64, num_block=20, num_points=9),
        # BasicUniVSRFeatPropWithFlowDeformAtt_Fast_V2(num_feat=64, num_block=20, num_points=9)
    ]

    # b, n, c, h, w = 1, 10, 3, 720 // 4, 1280 // 4
    # b, n, c, h, w = 1, 10, 3, 1080 // 4, 1920 // 4
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
