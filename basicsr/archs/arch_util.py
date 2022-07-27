import collections.abc
import math
import time
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from einops import rearrange
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair, _single
from torch.utils.checkpoint import checkpoint

from basicsr.ops.cot import \
    CotLayer, CoXtLayer, CotLayerNoNorm, CoXtLayerNoNorm, CotLayerWithLayerNorm, CoXtLayerWithLayerNorm
from basicsr.ops.dcn import \
    ModulatedDeformConvPack, ModulatedDeformConv, modulated_deform_conv
from basicsr.ops.msda import \
    MSDeformAttn, MSDeformAttnMyVersion, SingleScaleDeformAttnV1, SingleScaleDeformAttnV2, SingleScaleDeformAttnV3
from basicsr.utils import get_root_logger


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ConvReLU(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size, stride, padding, has_relu=True, efficient=False):
        super(ConvReLU, self).__init__()
        self.has_relu = has_relu
        self.efficient = efficient

        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        def _func_factory(conv, relu, has_relu):
            def func(x):
                x = conv(x)
                if has_relu:
                    x = relu(x)
                return x

            return func

        func = _func_factory(self.conv, self.relu, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x


class ConvNextResBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=3, exp_ratio=4, act_type='relu'):
        super(ConvNextResBlock, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type

        self.dwconv = nn.Conv2d(inp_channels, inp_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv1x1_0 = nn.Conv2d(inp_channels, out_channels * exp_ratio, 1)
        self.conv1x1_1 = nn.Conv2d(out_channels * exp_ratio, out_channels, 1)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.dwconv(x)
        y = self.conv1x1_0(y)
        y = self.act(y)
        y = self.conv1x1_1(y) + x
        return y


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, num_feat=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, min_channels=8, reduction_channels=None):
        super(SEModule, self).__init__()
        reduction_channels = reduction_channels or max(channels // reduction, min_channels)
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class CoTResBottleneck(nn.Module):

    def __init__(self, inplanes, planes, expansion=1, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(CoTResBottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        if stride > 1:
            self.avd = nn.AvgPool2d(3, 2, padding=1)
        else:
            self.avd = None

        self.conv2 = CotLayer(width, kernel_size=3) if cardinality == 1 else CoXtLayer(width, kernel_size=3)

        # self.conv2 = nn.Conv2d(
        #    first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
        #    padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        # self.bn2 = norm_layer(width)
        # self.act2 = act_layer(inplace=True)
        # self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = SEModule(channels=outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        if self.avd is not None:
            x = self.avd(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        # if self.drop_block is not None:
        #    x = self.drop_block(x)
        # x = self.act2(x)
        # if self.aa is not None:
        #    x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)

        return x


class CoTResidualBlock(nn.Module):

    def __init__(self, num_feat=64, res_scale=1, norm=True):
        super(CoTResidualBlock, self).__init__()
        self.res_scale = res_scale
        if norm:
            self.conv1 = CotLayerWithLayerNorm(dim=num_feat, kernel_size=3)
            self.conv2 = CotLayerWithLayerNorm(dim=num_feat, kernel_size=3)
        else:
            self.conv1 = CotLayerNoNorm(dim=num_feat, kernel_size=3)
            self.conv2 = CotLayerNoNorm(dim=num_feat, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2
        )
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def flow_warp_sequence_v1(seq, flows_forward, flows_backward):
    """ Aligned sequences with

    Args:
    seq (Tensor): Input sequence with shape (b, n, c, h, w)
    flows_forward (Tensor): Forward flow sequence with shape (b, n-1, 2, h, w)
    flows_backward (Tensor): Backward flow sequence with shape (b, n-1, 2, h, w)
    """
    b, n, c, h, w = seq.size()

    prev_frames = seq[:, :-2, :, :, :]  # (b, n-2, c, h, w)
    next_frames = seq[:, 2:, :, :, :]  # (b, n-2, c, h, w)

    prev_flows = flows_backward[:, :-1, :, :, :]  # (b, n-2, 2, h, w)
    next_flows = flows_forward[:, 1:, :, :, :]  # (b, n-2, 2, h, w)

    # warped_prev_frames = []
    # warped_next_frames = []
    # # warped prev frames
    # for i in range(n - 2):
    #     warped_prev_frames.append(
    #         flow_warp(prev_frames[:, i, :, :, :], prev_flows[:, i, :, :, :].permute(0, 2, 3, 1))
    #     )
    # # warped next frames
    # for i in range(n - 2):
    #     warped_next_frames.append(
    #         flow_warp(next_frames[:, i, :, :, :], next_flows[:, i, :, :, :].permute(0, 2, 3, 1))
    #     )
    # warped_prev_frames = torch.stack(warped_prev_frames, dim=1)
    # warped_next_frames = torch.stack(warped_next_frames, dim=1)

    warped_prev_frames = flow_warp(prev_frames.view(-1, c, h, w),
                                   prev_flows.view(-1, 2, h, w).permute(0, 2, 3, 1)).view(b, n - 2, c, h, w)
    warped_next_frames = flow_warp(next_frames.view(-1, c, h, w),
                                   next_flows.view(-1, 2, h, w).permute(0, 2, 3, 1)).view(b, n - 2, c, h, w)

    # (b, n-2, c, h, w)
    return warped_prev_frames, warped_next_frames


def flow_warp_sequence_v2(seq, flows_forward, flows_backward):
    """ Aligned sequences with

    Args:
    seq (Tensor): Input sequence with shape (b, n, c, h, w)
    flows_forward (Tensor): Forward flow sequence with shape (b, n-1, 2, h, w)
    flows_backward (Tensor): Backward flow sequence with shape (b, n-1, 2, h, w)
    """
    b, n, c, h, w = seq.size()

    prev_frames = seq[:, :-1, :, :, :]  # (b, n-1, c, h, w)
    next_frames = seq[:, 1:, :, :, :]  # (b, n-1, c, h, w)

    warped_prev_frames = []
    warped_next_frames = []
    # warped prev frames
    for i in range(n - 1):
        warped_prev_frames.append(
            flow_warp(prev_frames[:, i, :, :, :], flows_backward[:, i, :, :, :].permute(0, 2, 3, 1))
        )
    # warped next frames
    for i in range(n - 1):
        warped_next_frames.append(
            flow_warp(next_frames[:, i, :, :, :], flows_forward[:, i, :, :, :].permute(0, 2, 3, 1))
        )
    warped_prev_frames = torch.stack(warped_prev_frames, dim=1)
    warped_next_frames = torch.stack(warped_next_frames, dim=1)

    # warped_prev_frames = flow_warp(prev_frames.view(-1, c, h, w),
    #                                prev_flows.view(-1, 2, h, w).permute(0, 2, 3, 1)).view(b, n - 1, c, h, w)
    # warped_next_frames = flow_warp(next_frames.view(-1, c, h, w),
    #                                next_flows.view(-1, 2, h, w).permute(0, 2, 3, 1)).view(b, n - 1, c, h, w)

    # (b, n-1, c, h, w)
    return warped_prev_frames, warped_next_frames


def flow_warp_sequence_v3(seq, flows_forward, flows_backward):
    """ Aligned sequences with

    Args:
    seq (Tensor): Input sequence with shape (b, n, c, h, w)
    flows_forward (Tensor): Forward flow sequence with shape (b, n-1, 2, h, w)
    flows_backward (Tensor): Backward flow sequence with shape (b, n-1, 2, h, w)
    """
    b, n, c, h, w = seq.size()

    prev_frames = seq[:, :-1, :, :, :]  # (b, n-1, c, h, w)
    next_frames = seq[:, 1:, :, :, :]  # (b, n-1, c, h, w)

    warped_prev_frames = []
    warped_next_frames = []
    # warped prev frames
    for i in range(n - 1):
        warped_prev_frames.append(
            flow_warp(prev_frames[:, i, :, :, :], flows_backward[:, i, :, :, :].permute(0, 2, 3, 1))
        )
    # warped next frames
    for i in range(n - 1):
        warped_next_frames.append(
            flow_warp(next_frames[:, i, :, :, :], flows_forward[:, i, :, :, :].permute(0, 2, 3, 1))
        )
    warped_prev_frames.insert(0, seq[:, 0, :, :, :])
    warped_next_frames.insert(n, seq[:, -1, :, :, :])
    warped_prev_frames = torch.stack(warped_prev_frames, dim=1)
    warped_next_frames = torch.stack(warped_next_frames, dim=1)

    # (b, n, c, h, w)
    return warped_prev_frames, warped_next_frames


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def bilinear_sampler(img, flow_s2r_lvl, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = flow_s2r_lvl.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    # img = F.grid_sample(img, grid, align_corners=True)
    img = F.grid_sample(img, grid, align_corners=True)  # check: align_corners

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


class FGAC(nn.Module):
    def __init__(self, nf):
        super(FGAC, self).__init__()
        """ Flow-Guided Attentive Correlation """
        self.visualization_flag = False
        self.nf = nf
        self.scale = [1]

        self.conv_ref_k = nn.Conv2d(self.nf, self.nf, [1, 1], 1, [0, 0])
        self.conv_source_k = nn.Conv2d(self.nf, self.nf, [1, 1], 1, [0, 0])
        self.feature_ch = self.nf
        self.softmax = nn.Softmax(dim=1)

        self.w_gen = nn.Conv2d(self.nf * 2, self.nf, [3, 3], 1, [1, 1])
        self.w_gen_2 = nn.Conv2d(self.nf, 1, [3, 3], 1, [1, 1])
        self.relu = nn.ReLU()

        self.fusion = nn.Conv2d(self.nf, self.nf, [1, 1], 1, [0, 0])

        # self.w = torch.tensor([1.0], requires_grad=True, device=device)
        # optimizer = torch.optim.Adam([{'params':model_net.parameters()},
        # 							  {'params':model_net.FAC_FB_Module.FGAC_F1toF0.w,'lr':1e-3},
        # 							  {'params':model_net.FAC_FB_Module.FGAC_F0toF1.w,'lr':1e-3}], lr=args.init_lr,
        # 							 betas=(0.9, 0.999), weight_decay=args.weight_decay)  # optimizer in "main.py"

    def forward(self, ref, source, flow_s2r):
        init_ref_k = self.conv_ref_k(ref)
        init_source_k = self.conv_source_k(source)
        source_v = source

        ref_k = init_ref_k
        source_k = init_source_k

        flow_s2r = flow_s2r.contiguous().permute(0, 2, 3, 1).float()  # [B,H,W,2]
        f_bs, f_h, f_w, f_c = flow_s2r.shape

        """ 
            This is a generalized version when there are both radii for sources (sr) and ref. (rr) 
            For DeMFI, due to point-wise FGAC, we set rr=0 and sr=0.            
        """
        rr = 0
        sr = 0
        """ (i) make centroid based on flow_s2r, then bilinear sampling on ref_k """
        # (i-1): make grid
        dx = torch.linspace(-rr, rr, 2 * rr + 1)
        dy = torch.linspace(-rr, rr, 2 * rr + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(flow_s2r.device)  # [B,2rr+1,2rr+1,2]
        delta_lvl = delta.contiguous().view(1, 1, 2 * rr + 1, 1, 2 * rr + 1, 2).repeat(1, f_h, 1, f_w, 1, 1). \
            contiguous().view(1, f_h * (2 * rr + 1), f_w * (2 * rr + 1), 2)  # [B, H*(2rr+1),W*(2rr+1),2]

        # (i-2): make centroid by using flow
        # flow_s2r = flow_s2r.contiguous().view(1, 1,f_h, 1,f_w, 2).repeat(1, 2*rr+1, 1, 2*rr+1, 1, 1)
        centroid_lvl = flow_s2r.repeat(1, 2 * rr + 1, 2 * rr + 1, 1)  # [B,H*(2rr+1),W*(2rr+1),2]

        # (i-3): make flow-grid and bilinear sampling
        flow_s2r_lvl = centroid_lvl + delta_lvl  # grid (including flow and coordinates): [B,H*(2rr+1),W*(2rr+1), 2]
        ref_k = F.avg_pool2d(ref_k, (2 * sr + 1, 2 * sr + 1), (1, 1), padding=sr)
        # gathering size of "source grid" in ref_k via average pooling.
        indexed_ref_k = bilinear_sampler(ref_k, flow_s2r_lvl)  # ref: [B,c,h,w], grid: [B,H*(2rr+1),W*(2rr+1), 2]
        # indexed_ref_k: [B,C,H*(2rr+1),W*(2rr+1)] (following dim. of grid)

        indexed_ref_k = indexed_ref_k.contiguous().view(f_bs, self.feature_ch, f_h, (2 * rr + 1), f_w,
                                                        (2 * rr + 1)).permute(0, 1, 3, 2, 5, 4)
        indexed_ref_k = indexed_ref_k.contiguous().view(f_bs, self.feature_ch, (2 * rr + 1) * f_h,
                                                        (2 * rr + 1) * f_w)  # [batch,C,(2rr+1)*H,(2rr+1)*W]
        # caution: order is very important !
        indexed_ref_k = F.unfold(indexed_ref_k,
                                 kernel_size=((2 * rr + 1), (2 * rr + 1)),
                                 stride=((2 * rr + 1), (2 * rr + 1)), padding=rr)  # [batch, C*((2rr+1)**2), H, W]
        grid_sampled_ref_k = indexed_ref_k.contiguous().view(f_bs, self.feature_ch, (2 * rr + 1) ** 2, f_h, f_w)
        # [batch, C, (2rr+1)**2, H, W]

        """ (ii) unfold source_k for computing attentive correlation """
        source_k = F.avg_pool2d(source_k, (2 * sr + 1, 2 * sr + 1), (1, 1), padding=sr)
        # gathering size of "source grid" in source_k via average pooling.
        source_k = torch.unsqueeze(source_k, 2)
        # [batch, C, 1, H, W]
        corr_r2s_k = torch.sum(grid_sampled_ref_k * source_k, 1)  # ab
        # element-wise multiplication (source_k is broadcasted), then sum.
        # [batch, (2rr+1)**2, H, W]
        softmax_corr_r2s_k = torch.unsqueeze(self.softmax(corr_r2s_k), 1)
        # [batch, 1, (2rr+1)**2, H, W]
        FAC_sr = torch.sum(grid_sampled_ref_k * softmax_corr_r2s_k, 2)  # Eq.(3)
        # element-wise multiplication (softmax_corr_r2s_k is broadcasted)
        # [batch, C, H, W]

        E_s = self.fusion(FAC_sr)  # right term of Eq.(4)
        w_sr = torch.sigmoid(self.w_gen_2(
            self.relu(self.w_gen(torch.cat([source_v, E_s], dim=1)))))  # spatially variant (adaptive)

        bolstered_F_s = w_sr * source_v + (1 - w_sr) * E_s  # Eq.(4)

        """ min-max normalization for visualization of difference feature maps after applying Eq.(4) """
        # diff = torch.abs(bolstered_F_s) - torch.abs(source_v)
        diff = bolstered_F_s - source_v
        diff = torch.mean(torch.abs(diff), 1, keepdim=True)
        b, c, h, w = diff.shape
        diff = diff.view(b, -1)
        diff -= diff.min(1, keepdim=True)[0]
        diff /= diff.max(1, keepdim=True)[0]
        diff = diff.view(b, 1, h, w)

        if self.visualization_flag:
            E_s = torch.mean(torch.abs(E_s), 1, keepdim=True)
            b, c, h, w = E_s.shape
            E_s = E_s.view(b, -1)
            E_s -= E_s.min(1, keepdim=True)[0]
            E_s /= E_s.max(1, keepdim=True)[0]
            E_s = E_s.view(b, 1, h, w)

            source_v = torch.mean(torch.abs(source_v), 1, keepdim=True)
            b, c, h, w = source_v.shape
            source_v = source_v.view(b, -1)
            source_v -= source_v.min(1, keepdim=True)[0]
            source_v /= source_v.max(1, keepdim=True)[0]
            source_v = source_v.view(b, 1, h, w)

            init_ref_k = torch.mean(torch.abs(init_ref_k), 1, keepdim=True)
            b, c, h, w = init_ref_k.shape
            init_ref_k = init_ref_k.view(b, -1)
            init_ref_k -= init_ref_k.min(1, keepdim=True)[0]
            init_ref_k /= init_ref_k.max(1, keepdim=True)[0]
            init_ref_k = init_ref_k.view(b, 1, h, w)

            bolstered_F_s_ch1 = torch.mean(torch.abs(bolstered_F_s), 1, keepdim=True)
            b, c, h, w = bolstered_F_s_ch1.shape
            bolstered_F_s_ch1 = bolstered_F_s_ch1.view(b, -1)
            bolstered_F_s_ch1 -= bolstered_F_s_ch1.min(1, keepdim=True)[0]
            bolstered_F_s_ch1 /= bolstered_F_s_ch1.max(1, keepdim=True)[0]
            bolstered_F_s_ch1 = bolstered_F_s_ch1.view(b, 1, h, w)

            return bolstered_F_s, [w_sr, (1 - w_sr), source_v, init_ref_k, E_s, bolstered_F_s_ch1], diff
        else:
            return bolstered_F_s, w_sr, diff


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


class IterativeAlignmentUnit(nn.Module):
    """Iterative Alignment Unit.

    Args:
        num_feat (int): Channel number of intermediate features. Default: 64.
    """

    def __init__(self, num_feat=64, kernel_size=(3, 3), deformable_groups=8):
        super(IterativeAlignmentUnit, self).__init__()

        offset_dim = deformable_groups * 2 * kernel_size[0] * kernel_size[1]
        mask_dim = deformable_groups * 1 * kernel_size[0] * kernel_size[1]

        self.offset1_module_1 = ConvReLU(num_feat * 2, num_feat, 3, 1, 1, has_relu=True)
        self.offset1_module_2 = nn.Conv2d(num_feat, offset_dim + mask_dim, 3, 1, 1)

        self.offset2_module_1 = ConvReLU((offset_dim + mask_dim) * 2, num_feat, 3, 1, 1, has_relu=True)
        self.offset2_module_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.offset2_module_3 = nn.Conv2d(num_feat, offset_dim + mask_dim, 3, 1, 1)

        self.modulated_deform_conv = ModulatedDeformConv(num_feat, num_feat, 3, 1, 1)

    def forward(self, nbr_fea, ref_fea, offset_prev=None):

        fea = torch.cat([nbr_fea, ref_fea], dim=1)
        offset_mask = self.offset1_module_2(self.offset1_module_1(fea))
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset_curr = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        if offset_prev:
            offset_curr = torch.cat([offset_prev, offset_mask], dim=1)
            offset_curr = self.offset2_module_1(offset_curr)
            offset_curr = self.offset2_module_2(offset_curr)
            offset_mask = self.offset2_module_3(offset_curr)
            o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
            offset_curr = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)

        aligned_fea = self.modulated_deform_conv(nbr_fea, offset_curr, mask)

        return aligned_fea, offset_curr


class FirstOrderDeformableAlignment(ModulatedDeformConvPack):
    """First-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(FirstOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.groups, 3, 1, 1),
        )

        # self.conv_offset = nn.Sequential(
        #     nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Conv2d(self.out_channels, 27 * self.groups, 3, 1, 1),
        # )

        # self.conv_offset = nn.Sequential(
        #     nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Conv2d(self.out_channels, 27 * self.groups, 3, 1, 1),
        # )

        # self.conv_offset = nn.Sequential(
        #     nn.Conv2d(2 * self.out_channels + 2, 27 * self.groups, 3, 1, 1)
        # )

        # self.conv_output = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow):
        extra_feat = torch.cat([extra_feat, flow], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset + flow
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding,
                                     self.dilation, self.groups,
                                     self.groups)
        # return self.conv_output(x)


class BwFlowGuidedFirstOrderDeformableAlignment(ModulatedDeformConvPack):
    """First-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(BwFlowGuidedFirstOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow):
        extra_feat = torch.cat([extra_feat, flow], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset + flow
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding,
                                     self.dilation, self.groups,
                                     self.groups)


class FwFlowGuidedFirstOrderDeformableAlignment(ModulatedDeformConvPack):
    """First-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(FwFlowGuidedFirstOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow):
        extra_feat = torch.cat([extra_feat, flow], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset + flow
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset - flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding,
                                     self.dilation, self.groups,
                                     self.groups)


class FirstOrderDeformableAlignmentV1(ModulatedDeformConvPack):
    """First-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(FirstOrderDeformableAlignmentV1, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow):
        extra_feat = torch.cat([extra_feat, flow], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset + flow
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding,
                                     self.dilation, self.groups,
                                     self.groups)


class FirstOrderDeformableAlignmentV2(ModulatedDeformConvPack):
    """First-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(FirstOrderDeformableAlignmentV2, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow):

        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset + flow
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)
        output = modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.groups)

        return output


class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding,
                                     self.dilation, self.groups,
                                     self.groups)


class MultiscaleDeformAttnAlign(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ms_deform_att = MSDeformAttnMyVersion(d_model=d_model,
                                                   n_levels=n_levels,
                                                   n_heads=n_heads,
                                                   n_points=n_points,
                                                   max_residue_magnitude=max_residue_magnitude)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, nbr_fea, ext_fea):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        mask = (torch.zeros(b, h, w) > 1).to(device)

        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratio = torch.unsqueeze(self.get_valid_ratio(mask), dim=1)
        ref_point = self.get_reference_points(spatial_shapes, valid_ratio, device=device)

        output = self.ms_deform_att(ext_fea, ref_point, nbr_fea, spatial_shapes, level_start_index,
                                    input_padding_mask=mask.flatten(1), flow=None)

        return output


class FlowGuidedDeformAttnAlignV1(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ms_deform_att = MSDeformAttnMyVersion(d_model=d_model,
                                                   n_levels=n_levels,
                                                   n_heads=n_heads,
                                                   n_points=n_points,
                                                   max_residue_magnitude=max_residue_magnitude)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, nbr_fea, ext_fea, flow):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        mask = (torch.zeros(b, h, w) > 1).to(device)

        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratio = torch.unsqueeze(self.get_valid_ratio(mask), dim=1)
        ref_point = self.get_reference_points(spatial_shapes, valid_ratio, device=device)

        output = self.ms_deform_att(ext_fea, ref_point, nbr_fea, spatial_shapes, level_start_index,
                                    input_padding_mask=mask.flatten(1), flow=flow)

        return output


class FlowGuidedDeformAttnAlignV2(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ms_deform_att = SingleScaleDeformAttnV1(d_model=d_model,
                                                     n_heads=n_heads,
                                                     n_points=n_points,
                                                     max_residue_magnitude=max_residue_magnitude)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, nbr_fea, ext_fea, flow):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        mask = (torch.zeros(b, h, w) > 1).to(device)

        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratio = torch.unsqueeze(self.get_valid_ratio(mask), dim=1)
        ref_point = self.get_reference_points(spatial_shapes, valid_ratio, device=device)

        output = self.ms_deform_att(ext_fea, ref_point, nbr_fea, spatial_shapes, level_start_index,
                                    input_padding_mask=mask.flatten(1), flow=flow)

        return output


class FlowGuidedDeformAttnAlignV3(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ms_deform_att = SingleScaleDeformAttnV2(d_model=d_model,
                                                     n_heads=n_heads,
                                                     n_points=n_points,
                                                     max_residue_magnitude=max_residue_magnitude)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, nbr_fea, ext_fea, flow):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        mask = (torch.zeros(b, h, w) > 1).to(device)

        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratio = torch.unsqueeze(self.get_valid_ratio(mask), dim=1)
        ref_point = self.get_reference_points(spatial_shapes, valid_ratio, device=device)

        output = self.ms_deform_att(ext_fea, ref_point, nbr_fea, spatial_shapes, level_start_index,
                                    input_padding_mask=mask.flatten(1), flow=flow)

        return output


class FlowGuidedDeformAttnAlignV4(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ss_deform_att = SingleScaleDeformAttnV3(d_model=d_model,
                                                     n_heads=n_heads,
                                                     n_points=n_points,
                                                     max_residue_magnitude=max_residue_magnitude)

    def get_ref_points(self, H_, W_, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
        )
        ref = torch.stack((ref_x, ref_y), -1)
        ref[..., 0].div_(W_)  #.mul_(2).sub_(1)
        ref[..., 1].div_(H_)  #.mul_(2).sub_(1)
        ref = ref[None, ...]

        return ref

    def forward(self, nbr_fea, ext_fea, flow):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        ref_point = self.get_ref_points(h, w, device=device)
        output = self.ss_deform_att(ext_fea, nbr_fea, ref_point, flow=flow)

        return output


class FlowGuidedDeformConvInterpolation(nn.Module):

    def __init__(self, inp_channels=64, out_channels=64, deformable_groups=8, max_residue_magnitude=10):
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.groups = deformable_groups
        self.max_residue_magnitude = max_residue_magnitude

        self.modulated_deform_conv_1 = ModulatedDeformConv(inp_channels, out_channels, 3,
                                                           stride=1, padding=1,
                                                           dilation=1, groups=1,
                                                           deformable_groups=self.groups, bias=True)
        self.modulated_deform_conv_2 = ModulatedDeformConv(inp_channels, out_channels, 3,
                                                           stride=1, padding=1,
                                                           dilation=1, groups=1,
                                                           deformable_groups=self.groups, bias=True)

        self.conv_offset_1 = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.groups, 3, 1, 1),
        )
        self.conv_offset_2 = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.groups, 3, 1, 1),
        )
        self.fusion_conv = nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1)

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset_1[-1], val=0, bias=0)
        constant_init(self.conv_offset_2[-1], val=0, bias=0)

    def forward(self, feat_prev, feat_next, extra_feat_prev, extra_feat_next, flow_prev, flow_next):
        out = self.conv_offset_1(extra_feat_prev)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        # offset + flow
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow_prev.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        # mask
        mask = torch.sigmoid(mask)
        feat_prev = self.modulated_deform_conv_1(feat_prev, offset, mask)

        out = self.conv_offset_2(extra_feat_next)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        # offset + flow
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow_next.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        # mask
        mask = torch.sigmoid(mask)
        feat_next = self.modulated_deform_conv_2(feat_next, offset, mask)

        feat_curr = self.fusion_conv(torch.cat([feat_prev, feat_next], dim=1))

        return feat_curr


class FlowGuidedDeformAttnInterpolation(nn.Module):

    def __init__(self, inp_channels=64, out_channels=64, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.n_points = n_points
        self.max_residue_magnitude = max_residue_magnitude
        self.ss_deform_att_1 = SingleScaleDeformAttnV1(d_model=inp_channels, n_heads=n_heads, n_points=n_points,
                                                       max_residue_magnitude=max_residue_magnitude)
        self.ss_deform_att_2 = SingleScaleDeformAttnV1(d_model=inp_channels, n_heads=n_heads, n_points=n_points,
                                                       max_residue_magnitude=max_residue_magnitude)
        self.fusion_conv = nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, feat_prev, feat_next, extra_feat_prev, extra_feat_next, flow_prev, flow_next):
        b, c, h, w = feat_prev.shape
        device = feat_prev.device

        mask = (torch.zeros(b, h, w) > 1).to(device)
        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
        valid_ratio = torch.unsqueeze(self.get_valid_ratio(mask), dim=1)
        ref_point = self.get_reference_points(spatial_shapes, valid_ratio, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_prev = self.ss_deform_att_1(extra_feat_prev, ref_point, feat_prev, spatial_shapes, level_start_index,
                                         input_padding_mask=mask.flatten(1), flow=flow_prev)
        feat_next = self.ss_deform_att_2(extra_feat_next, ref_point, feat_next, spatial_shapes, level_start_index,
                                         input_padding_mask=mask.flatten(1), flow=flow_next)
        feat_curr = self.fusion_conv(torch.cat([feat_prev, feat_next], dim=1))

        return feat_curr


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


if __name__ == '__main__':
    device = torch.device('cuda')

    # inp = torch.randn(4, 32, 128, 128).to(device)
    # net = make_layer(CoTResBottleneck, 10, inplanes=32, planes=32).to(device)
    # out = net(inp)
    # print(out.shape)

    inp1 = torch.randn(1, 32, 64, 64).to(device)
    inp2 = torch.randn(1, 64, 64, 64).to(device)
    flow = torch.randn(1, 2, 64, 64).to(device)
    att = FirstOrderDeformableAlignmentV2(32, 32, 3,
                                          stride=1,
                                          padding=1,
                                          dilation=1,
                                          groups=1,
                                          deformable_groups=8,
                                          bias=True).to(device)
    for i in range(10):
        out = att(inp1, inp2, flow)

    # interpolator = FlowGuidedDeformConvInterpolation(inp_channels=32, out_channels=32, deformable_groups=8, max_residue_magnitude=10).to(device)
    # interpolator = FlowGuidedDeformAttnInterpolation(inp_channels=32, out_channels=32).to(device)
    #
    # inp1 = torch.randn(1, 32, 64, 64).to(device)
    # inp2 = torch.randn(1, 32, 64, 64).to(device)
    # ext1 = torch.randn(1, 64, 64, 64).to(device)
    # ext2 = torch.randn(1, 64, 64, 64).to(device)
    # flow1 = torch.randn(1, 2, 64, 64).to(device)
    # flow2 = torch.randn(1, 2, 64, 64).to(device)
    #
    # out = interpolator(inp1, inp2, ext1, ext2, flow1, flow2)
    # print(out.shape)
