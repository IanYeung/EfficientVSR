import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.ops.cot import \
    CotLayer, CoXtLayer, CotLayerNoNorm, CoXtLayerNoNorm, CotLayerWithLayerNorm, CoXtLayerWithLayerNorm
from .arch_util import \
    ResidualBlockNoBN, ConvNextResBlock, flow_warp, make_layer, \
    FirstOrderDeformableAlignment, FirstOrderDeformableAlignmentV2, \
    FlowGuidedDeformAttnAlignV1, FlowGuidedDeformAttnAlignV2, \
    FlowGuidedDeformAttnAlignV3, FlowGuidedDeformAttnAlignV4, \
    FGAC
from .edvr_arch import PCDAlignment
from .rcan_arch import RCAB
from .elan_arch import LFE
from .imdn_arch import IMDM
from .iavr_arch import LocalCorr
from .raft_arch import RAFT_SR
from .spynet_arch import SpyNet
from .pwcnet_arch import PWCNet
from .fastflownet_arch import FastFlowNet
from .maskflownet_arch import MaskFlownet_S


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15, act=nn.LeakyReLU(negative_slope=0.1, inplace=True)):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            act,
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch)
        )

    def forward(self, fea):
        return self.main(fea)


# @ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFastFlowDCN_Test(nn.Module):
    """Online VSR with Flow Guided Deformable Alignment and ResidualNoBN Reconstuction Branch.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 deformable_groups=8,
                 flownet_path=None,
                 return_flow=False,
                 one_stage_up=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow
        self.one_stage_up = one_stage_up

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.flownet = FastFlowNet(groups=3, load_path=flownet_path)
        self.flow_guided_dcn = FirstOrderDeformableAlignment(num_feat,
                                                             num_feat,
                                                             3,
                                                             stride=1,
                                                             padding=1,
                                                             dilation=1,
                                                             groups=1,
                                                             deformable_groups=deformable_groups,
                                                             bias=True)

        # propagation
        self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

        # reconstruction
        if self.one_stage_up:
            self.upconv = nn.Conv2d(num_feat, 3 * 16, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(4)
        else:
            self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, frm_curr, frm_prev, feat_prop, init=False):

        flow = self.flownet(frm_curr, frm_prev)
        feat_curr = self.feat_extract(frm_curr)

        if not init:
            extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
            feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)

        feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
        feat_prop = self.forward_trunk(feat_prop)

        # upsample
        out = feat_prop
        if self.one_stage_up:
            out = self.pixel_shuffle(self.upconv(out))
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
        base = F.interpolate(frm_curr, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        return out, feat_prop