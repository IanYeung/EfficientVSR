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

from .timesformer_arch import TemporalAttention
from .vswintransformer_arch import BasicLayer as VideoSwinAttention
from .sepststransformer_arch import SepSTSBasicLayer as SepSTSAttention
from .swinir_fast_arch import SelfAttBlock, CrossAttBlock, LayerNorm
from .shuffle_transformer_arch import Block as ShuffleTransformerBlock


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


class ConvShiftConvBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(LFE, num_block, inp_channels=num_out_ch, out_channels=num_out_ch, exp_ratio=1)
        )

    def forward(self, fea):
        return self.main(fea)


class ConvResChAttBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(RCAB, num_block, num_feat=num_out_ch)
        )

    def forward(self, fea):
        return self.main(fea)


class ConvIMDBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(IMDM, num_block, in_channels=num_out_ch),
        )

    # def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15, num_mid_ch=128):
    #     super().__init__()
    #     self.main = nn.Sequential(
    #         nn.Conv2d(num_in_ch, num_mid_ch, 3, 1, 1, bias=True),
    #         nn.LeakyReLU(negative_slope=0.1, inplace=True),
    #         make_layer(IMDM, num_block, in_channels=num_mid_ch),
    #         nn.Conv2d(num_mid_ch, num_out_ch, 3, 1, 1, bias=True),
    #         nn.LeakyReLU(negative_slope=0.1, inplace=True)
    #     )

    def forward(self, fea):
        return self.main(fea)


class ConvConvNextBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15, kernel_size=3, exp_ratio=2, act_type='relu'):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(
                ConvNextResBlock,
                num_block,
                inp_channels=num_out_ch,
                out_channels=num_out_ch,
                kernel_size=kernel_size,
                exp_ratio=exp_ratio,
                act_type=act_type
            )
        )

    def forward(self, fea):
        return self.main(fea)


# ablation on feature alignment module
@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithoutAlign_Fast(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 one_stage_up=False):
        super().__init__()
        self.num_feat = num_feat
        self.one_stage_up = one_stage_up

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

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

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """

        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]

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
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithSpyFlow(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 spynet_path=None,
                 return_flow=False,
                 one_stage_up=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow
        self.one_stage_up = one_stage_up

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)

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

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

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
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithSpyFlow_FGAC(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 spynet_path=None,
                 return_flow=False,
                 one_stage_up=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow
        self.one_stage_up = one_stage_up

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
        self.fgac = FGAC(nf=num_feat)

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

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                # feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                feat_prop, _, _ = self.fgac(feat_prop, feat_curr, flow)

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
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithSpyFlow_Fast(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 spynet_path=None,
                 return_flow=False,
                 one_stage_up=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow
        self.one_stage_up = one_stage_up

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)

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

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

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
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFastFlow_Fast(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
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

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

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
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithMaskFlow_Fast(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
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
        self.flownet = MaskFlownet_S(load_path=flownet_path)

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

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

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
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithRAFT_Fast(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 model='small',
                 flownet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.flownet = RAFT_SR(model=model, load_path=flownet_path)

        # propagation
        self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithPWCFlow_Fast(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 flownet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.flownet = PWCNet()

        # propagation
        self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithLocalCorr_Fast(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 nbr_size=11):
        super().__init__()
        self.num_feat = num_feat

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.localcorr = LocalCorr(nf=num_feat, nbr_size=nbr_size)

        # propagation
        self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                feat_prop = self.localcorr(feat_prop, feat_curr)

            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithPCDAlign(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

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
                 spynet_path=None,
                 return_flow=False,
                 one_stage_up=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow
        self.one_stage_up = one_stage_up

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # alignment
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)

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

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop_lv1 = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr_lv1 = feat[:, i, :, :, :]
            feat_curr_lv2 = self.lrelu(self.conv_l2_2(self.lrelu(self.conv_l2_1(feat_curr_lv1))))
            feat_curr_lv3 = self.lrelu(self.conv_l3_2(self.lrelu(self.conv_l3_1(feat_curr_lv2))))
            curr_feat_l = [feat_curr_lv1.clone(), feat_curr_lv2.clone(), feat_curr_lv3.clone()]

            if i > 0:
                feat_prop_lv1 = feat_prop_lv1
                feat_prop_lv2 = self.lrelu(self.conv_l2_2(self.lrelu(self.conv_l2_1(feat_prop_lv1))))
                feat_prop_lv3 = self.lrelu(self.conv_l3_2(self.lrelu(self.conv_l3_1(feat_prop_lv2))))
                prop_feat_l = [feat_prop_lv1.clone(), feat_prop_lv2.clone(), feat_prop_lv3.clone()]
                feat_prop_lv1 = self.pcd_align(prop_feat_l, curr_feat_l)

            feat_prop_lv1 = torch.cat([feat_curr_lv1, feat_prop_lv1], dim=1)
            feat_prop_lv1 = self.forward_trunk(feat_prop_lv1)

            # upsample
            out = feat_prop_lv1
            if self.one_stage_up:
                out = self.pixel_shuffle(self.upconv(out))
            else:
                out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
                out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFlowDCN(nn.Module):
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
                 spynet_path=None,
                 return_flow=False,
                 one_stage_up=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow
        self.one_stage_up = one_stage_up

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
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

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
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
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithSpyFlowDCN_Fast(nn.Module):
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
                 spynet_path=None,
                 return_flow=False,
                 one_stage_up=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow
        self.one_stage_up = one_stage_up

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
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

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
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
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFastFlowDCN_Fast(nn.Module):
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

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
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
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithMaskFlowDCN_Fast(nn.Module):
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
        self.flownet = MaskFlownet_S(load_path=flownet_path)
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

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
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
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFlowDeformAtt(nn.Module):
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
                 num_levels=1,
                 num_heads=8,
                 num_points=4,
                 max_residue_magnitude=10,
                 spynet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
        self.flow_guided_dcn = FlowGuidedDeformAttnAlign(d_model=num_feat,
                                                         n_levels=num_levels,
                                                         n_heads=num_heads,
                                                         n_points=num_points,
                                                         max_residue_magnitude=max_residue_magnitude)

        # propagation
        self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)

            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFlowDeformAtt_Fast_V1(nn.Module):
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
                 num_levels=1,
                 num_heads=8,
                 num_points=4,
                 max_residue_magnitude=10,
                 spynet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
        self.flow_guided_dcn = FlowGuidedDeformAttnAlignV1(d_model=num_feat,
                                                           n_levels=num_levels,
                                                           n_heads=num_heads,
                                                           n_points=num_points,
                                                           max_residue_magnitude=max_residue_magnitude)

        # propagation
        self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)

            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFlowDeformAtt_Fast_V2(nn.Module):
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
                 num_levels=1,
                 num_heads=8,
                 num_points=4,
                 max_residue_magnitude=10,
                 spynet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.layer_norm = LayerNorm(num_feat)
        self.spynet = SpyNet(spynet_path)
        self.flow_guided_dcn = FlowGuidedDeformAttnAlignV2(d_model=num_feat,
                                                           n_levels=num_levels,
                                                           n_heads=num_heads,
                                                           n_points=num_points,
                                                           max_residue_magnitude=max_residue_magnitude)

        # propagation
        self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                residual = feat_prop
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)
                feat_prop += residual
                feat_prop = self.layer_norm(feat_prop)

            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFlowDeformAtt_Fast_V3(nn.Module):
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
                 num_levels=1,
                 num_heads=8,
                 num_points=4,
                 max_residue_magnitude=10,
                 spynet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
        self.flow_guided_dcn = FlowGuidedDeformAttnAlignV3(d_model=num_feat,
                                                           n_levels=num_levels,
                                                           n_heads=num_heads,
                                                           n_points=num_points,
                                                           max_residue_magnitude=max_residue_magnitude)

        # propagation
        self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)

            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFlowDeformAtt_Fast_V4(nn.Module):
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
                 num_levels=1,
                 num_heads=8,
                 num_points=4,
                 max_residue_magnitude=10,
                 spynet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
        self.flow_guided_dcn = FlowGuidedDeformAttnAlignV4(d_model=num_feat,
                                                           n_levels=num_levels,
                                                           n_heads=num_heads,
                                                           n_points=num_points,
                                                           max_residue_magnitude=max_residue_magnitude)

        # propagation
        self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)

            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFlowDeformAtt_ShuffleAtt_Fast(nn.Module):
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
                 num_levels=1,
                 num_heads=8,
                 num_points=4,
                 window_size=8,
                 mlp_ratio=1,
                 max_residue_magnitude=10,
                 spynet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
        self.flow_guided_dcn = FlowGuidedDeformAttnAlignV2(d_model=num_feat,
                                                           n_levels=num_levels,
                                                           n_heads=num_heads,
                                                           n_points=num_points,
                                                           max_residue_magnitude=max_residue_magnitude)

        self.shuffle_blk1 = ShuffleTransformerBlock(num_feat, num_feat, num_heads, window_size=window_size,
                                                    shuffle=False, mlp_ratio=mlp_ratio)
        self.shuffle_blk2 = ShuffleTransformerBlock(num_feat, num_feat, num_heads, window_size=window_size,
                                                    shuffle=True, mlp_ratio=mlp_ratio)

        # propagation
        self.forward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)

            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


# ablation on feature propagation and feature fusion
@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropEarFusion(nn.Module):
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
                 spynet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
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
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)

            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropMidFusion(nn.Module):
    """Online VSR with Flow Guided Deformable Alignment and Dual ResidualNoBN Reconstuction Branch.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block_1=15,
                 num_block_2=15,
                 deformable_groups=8,
                 spynet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
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
        self.forward_trunk_1 = ConvResidualBlocks(1 * num_feat, num_feat, num_block_1)
        self.forward_trunk_2 = ConvResidualBlocks(2 * num_feat, num_feat, num_block_2)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            feat_curr = self.forward_trunk_1(feat_curr)
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)
            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk_2(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRDualFeatProp(nn.Module):
    """Online VSR with Flow Guided Deformable Alignment and Dual Feature Propagation.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block_1=15,
                 num_block_2=15,
                 deformable_groups=8,
                 spynet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
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
        self.forward_trunk_1 = ConvResidualBlocks(2 * num_feat, num_feat, num_block_1)
        self.forward_trunk_2 = ConvResidualBlocks(2 * num_feat, num_feat, num_block_2)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop_1 = x.new_zeros(b, self.num_feat, h, w)
        feat_prop_2 = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr_1 = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr_1, flow_warp(feat_prop_1, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop_1 = self.flow_guided_dcn(feat_prop_1, extra_feat, flow)
            feat_prop_1 = self.forward_trunk_1(torch.cat([feat_curr_1, feat_prop_1], dim=1))
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_prop_1, flow_warp(feat_prop_2, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop_2 = self.flow_guided_dcn(feat_prop_2, extra_feat, flow)
            feat_prop_2 = self.forward_trunk_2(torch.cat([feat_prop_1, feat_prop_2], dim=1))

            # upsample
            out = feat_prop_2
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFlowDCN_Fast_FrmInp1(nn.Module):
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
        # self.flownet = SpyNet(flownet_path)
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

    def get_frames(self, x, t=0, nf=3, f=7):
        index = np.array([t - nf // 2 + i for i in range(nf)])
        index = np.clip(index, 0, f - 1).tolist()
        it = x[:, index, :, :, :]
        return it

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
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
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicUniVSRFeatPropWithFlowDCN_Fast_FrmInp3(nn.Module):
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
        # self.flownet = SpyNet(flownet_path)
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
        self.forward_trunk = ConvResidualBlocks(4 * num_feat, num_feat, num_block)

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

    def get_frames(self, x, t=0, nf=3, f=7):
        index = np.array([t - nf // 2 + i for i in range(nf)])
        index = np.clip(index, 0, f - 1).tolist()
        it = x[:, index, :, :, :]
        return it

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward = self.get_flow(x)
        b, n, c, h, w = x.size()

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)

            stack_feat = self.get_frames(feat, t=i, nf=3, f=n).view(b, 3*self.num_feat, h, w)
            feat_prop = torch.cat([stack_feat, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            if self.one_stage_up:
                out = self.pixel_shuffle(self.upconv(out))
            else:
                out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
                out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward
        else:
            return torch.stack(out_l, dim=1)
