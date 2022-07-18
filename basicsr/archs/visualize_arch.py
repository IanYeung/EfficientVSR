import torch
from torch import nn as nn
from torch.nn import functional as F

from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.arch_util import \
    ResidualBlockNoBN, flow_warp, make_layer, FirstOrderDeformableAlignment, FirstOrderDeformableAlignmentV2, \
    FlowGuidedDeformAttnAlignV1, FlowGuidedDeformAttnAlignV2, FlowGuidedDeformAttnAlignV3, FlowGuidedDeformAttnAlignV4
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.pwcnet_arch import PWCNet
from basicsr.archs.fastflownet_arch import FastFlowNet


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
                 spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)

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
        feat_prop_l = []
        feat_curr_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop_l.append(feat_prop)
            feat_curr_l.append(feat_curr)
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

        # return torch.stack(out_l, dim=1), flows_forward
        return out_l, feat_curr_l, feat_prop_l, flows_forward


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
                 flownet_path=None):
        super().__init__()
        self.num_feat = num_feat

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
        feat_prop_l = []
        feat_curr_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop_l.append(feat_prop)
            feat_curr_l.append(feat_curr)
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

        # return torch.stack(out_l, dim=1), flows_forward
        return out_l, feat_curr_l, feat_prop_l, flows_forward


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
                 flownet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.flownet = FastFlowNet(groups=3, load_path=flownet_path)

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
        feat_prop_l = []
        feat_curr_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop_l.append(feat_prop)
            feat_curr_l.append(feat_curr)
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

        # return torch.stack(out_l, dim=1), flows_forward
        return out_l, feat_curr_l, feat_prop_l, flows_forward


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
                 spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
        self.flow_guided_dcn = FirstOrderDeformableAlignmentV2(num_feat,
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
        feat_prop_l = []
        feat_curr_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)
            feat_prop_l.append(feat_prop)
            feat_curr_l.append(feat_curr)
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

        # return torch.stack(out_l, dim=1), flows_forward
        return out_l, feat_curr_l, feat_prop_l, flows_forward


class BasicUniVSRFeatPropWithSpyFlowDCN_Fast_V2(nn.Module):
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
                 spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

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
        feat_prop_l = []
        feat_curr_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)
            feat_prop_l.append(feat_prop)
            feat_curr_l.append(feat_curr)
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

            # enhance the feature
            feat_prop = feat_prop + feat_curr

        # return torch.stack(out_l, dim=1), flows_forward
        return out_l, feat_curr_l, feat_prop_l, flows_forward


class BasicUniVSRFeatPropWithPWCFlowDCN_Fast(nn.Module):
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
                 flownet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.flownet = PWCNet()
        self.flow_guided_dcn = FirstOrderDeformableAlignmentV2(num_feat,
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
        feat_prop_l = []
        feat_curr_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)
            feat_prop_l.append(feat_prop)
            feat_curr_l.append(feat_curr)
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

        # return torch.stack(out_l, dim=1), flows_forward
        return out_l, feat_curr_l, feat_prop_l, flows_forward


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
                 flownet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.flownet = FastFlowNet(load_path=flownet_path)
        self.flow_guided_dcn = FirstOrderDeformableAlignmentV2(num_feat,
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
        feat_prop_l = []
        feat_curr_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_curr, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)
            feat_prop_l.append(feat_prop)
            feat_curr_l.append(feat_curr)
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

        # return torch.stack(out_l, dim=1), flows_forward
        return out_l, feat_curr_l, feat_prop_l, flows_forward


class BasicUniVSRFeatPropWithSpyFlow_Fast_NoResLearn(nn.Module):
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
                 spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)

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
        feat_prop_l = []
        feat_curr_l = []

        # forward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            feat_curr = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop_l.append(feat_prop)
            feat_curr_l.append(feat_curr)
            feat_prop = torch.cat([feat_curr, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = feat_prop
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            out_l.append(out)

        # return torch.stack(out_l, dim=1), flows_forward
        return out_l, feat_curr_l, feat_prop_l, flows_forward


class RealTimeBasicVSRCouplePropWithSpyNet(nn.Module):
    """BasicVSRCoupleProp V2.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15.
        keyframe_stride (int): Keyframe stride. Default: 5.
        temporal_padding (int): Temporal padding. Default: 2.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        edvr_path (str): Path to the pretrained EDVR model. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 spynet_path=None):
        super().__init__()

        self.num_feat = num_feat

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(3 * num_feat, num_feat, num_block)

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

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        b, n, c, h, w = x.size()

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            feat_i = feat[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([feat_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_i = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([feat_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return out_l, flows_forward, flows_backward


class RealTimeBasicVSRCouplePropWithPWCNet(nn.Module):
    """BasicVSRCoupleProp V2.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15.
        keyframe_stride (int): Keyframe stride. Default: 5.
        temporal_padding (int): Temporal padding. Default: 2.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        edvr_path (str): Path to the pretrained EDVR model. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 flownet_path=None):
        super().__init__()

        self.num_feat = num_feat

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.flownet = PWCNet()

        # propagation
        self.backward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(3 * num_feat, num_feat, num_block)

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

        flows_backward = self.flownet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        b, n, c, h, w = x.size()

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            feat_i = feat[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([feat_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_i = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([feat_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return out_l, flows_forward, flows_backward


class RealTimeBasicVSRCouplePropWithFastFlowNet(nn.Module):
    """BasicVSRCoupleProp V2.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15.
        keyframe_stride (int): Keyframe stride. Default: 5.
        temporal_padding (int): Temporal padding. Default: 2.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        edvr_path (str): Path to the pretrained EDVR model. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 flownet_path=None):
        super().__init__()

        self.num_feat = num_feat

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.flownet = FastFlowNet(load_path=flownet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(3 * num_feat, num_feat, num_block)

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

        flows_backward = self.flownet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        b, n, c, h, w = x.size()

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            feat_i = feat[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([feat_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_i = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([feat_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return out_l, flows_forward, flows_backward


class RealTimeBasicVSRCouplePropWithSpyNetDCN(nn.Module):
    """BasicVSRCoupleProp V3.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15.
        keyframe_stride (int): Keyframe stride. Default: 5.
        temporal_padding (int): Temporal padding. Default: 2.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        edvr_path (str): Path to the pretrained EDVR model. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 num_extract_block=0,
                 num_block=15,
                 deformable_groups=8,
                 spynet_path=None):
        super().__init__()

        self.num_feat = num_feat

        # feature extraction
        self.feat_extract = ConvResidualBlocks(3, num_feat, num_extract_block)

        # alignment
        self.spynet = SpyNet(spynet_path)
        self.flow_guided_dcn = FirstOrderDeformableAlignmentV2(num_feat,
                                                             num_feat,
                                                             3,
                                                             stride=1,
                                                             padding=1,
                                                             dilation=1,
                                                             groups=1,
                                                             deformable_groups=deformable_groups,
                                                             bias=True)

        # propagation
        self.backward_trunk = ConvResidualBlocks(2 * num_feat, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(3 * num_feat, num_feat, num_block)

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

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        b, n, c, h, w = x.size()

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)

        feat = self.feat_extract(x.view(-1, c, h, w)).view(b, n, -1, h, w)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            feat_i = feat[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                extra_feat = torch.cat([feat_i, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)
            feat_prop = torch.cat([feat_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_i = feat[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                extra_feat = torch.cat([feat_i, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
                feat_prop = self.flow_guided_dcn(feat_prop, extra_feat, flow)
            feat_prop = torch.cat([feat_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return out_l, flows_forward, flows_backward
