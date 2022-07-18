# try:
#     from model import common
# except ModuleNotFoundError:
#     import common

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def make_model(args, parent=False):
    return ELAN(args)


class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0 * g:1 * g, 1, 2] = 1.0
        mask[:, 1 * g:2 * g, 1, 0] = 1.0
        mask[:, 2 * g:3 * g, 2, 1] = 1.0
        mask[:, 3 * g:4 * g, 0, 1] = 1.0
        mask[:, 4 * g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
        return y


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0 * g:1 * g, 0, 1, 2] = 1.0  ## left
        self.weight[1 * g:2 * g, 0, 1, 0] = 1.0  ## right
        self.weight[2 * g:3 * g, 0, 2, 1] = 1.0  ## up
        self.weight[3 * g:4 * g, 0, 0, 1] = 1.0  ## down
        self.weight[4 * g:, 0, 1, 1] = 1.0  ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y)
        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y


class LFE(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='relu'):
        super(LFE, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels * exp_ratio)
        self.conv1 = ShiftConv2d(out_channels * exp_ratio, out_channels)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)
        return y


class GMSA(nn.Module):
    def __init__(self, channels, heads=8, shifts=4, window_size=8):
        super(GMSA, self).__init__()
        self.channels = channels
        self.heads = heads
        self.shifts = shifts
        self.window_size = window_size

        self.split_chns = [channels * 2 // 3, channels * 2 // 3, channels * 2 // 3]
        self.project_inp = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * 2, kernel_size=1),
            nn.BatchNorm2d(self.channels * 2)
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.project_inp(x)
        x0, x1, x2 = torch.split(x, self.split_chns, dim=1)
        wsize0 = self.window_size // 2 ** 2
        wsize1 = self.window_size // 2 ** 1
        wsize2 = self.window_size // 2 ** 0
        if self.shifts > 0:
            x0 = torch.roll(x0, shifts=(-wsize0 // 2, -wsize0 // 2), dims=(2, 3))
            x1 = torch.roll(x1, shifts=(-wsize1 // 2, -wsize1 // 2), dims=(2, 3))
            x2 = torch.roll(x2, shifts=(-wsize2 // 2, -wsize2 // 2), dims=(2, 3))
        b0, c0, h0, w0 = x0.shape
        b1, c1, h1, w1 = x1.shape
        b2, c2, h2, w2 = x2.shape

        q0, v0 = rearrange(
            x0, 'b (qv head c) (h dh) (w dw) -> qv (b h w) head (dh dw) c',
            qv=2, head=self.heads, dh=wsize0, dw=wsize0
        )
        q1, v1 = rearrange(
            x1, 'b (qv head c) (h dh) (w dw) -> qv (b h w) head (dh dw) c',
            qv=2, head=self.heads, dh=wsize1, dw=wsize1
        )
        q2, v2 = rearrange(
            x2, 'b (qv head c) (h dh) (w dw) -> qv (b h w) head (dh dw) c',
            qv=2, head=self.heads, dh=wsize2, dw=wsize2
        )

        atn0 = (q0 @ q0.transpose(-2, -1))
        atn0 = atn0.softmax(dim=-1)

        atn1 = (q1 @ q1.transpose(-2, -1))
        atn1 = atn1.softmax(dim=-1)

        atn2 = (q2 @ q2.transpose(-2, -1))
        atn2 = atn2.softmax(dim=-1)

        out0 = (atn0 @ v0)
        out0 = rearrange(
            out0, '(b h w) head (dh dw) c-> b (head c) (h dh) (w dw)',
            h=h0 // wsize0, w=w0 // wsize0, head=self.heads, dh=wsize0, dw=wsize0
        )

        out1 = (atn1 @ v1)
        out1 = rearrange(
            out1, '(b h w) head (dh dw) c-> b (head c) (h dh) (w dw)',
            h=h1 // wsize1, w=w1 // wsize1, head=self.heads, dh=wsize1, dw=wsize1
        )

        out2 = (atn2 @ v2)
        out2 = rearrange(
            out2, '(b h w) head (dh dw) c-> b (head c) (h dh) (w dw)',
            h=h2 // wsize2, w=w2 // wsize2, head=self.heads, dh=wsize2, dw=wsize2
        )
        if self.shifts > 0:
            out0 = torch.roll(out0, shifts=(wsize0 // 2, wsize0 // 2), dims=(2, 3))
            out1 = torch.roll(out1, shifts=(wsize1 // 2, wsize1 // 2), dims=(2, 3))
            out2 = torch.roll(out2, shifts=(wsize2 // 2, wsize2 // 2), dims=(2, 3))

        out = torch.cat((out0, out1, out2), dim=1)
        out = self.project_out(out)

        return out


class ELAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_size=8, heads=6):
        super(ELAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_size = window_size
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.gmsa = GMSA(channels=inp_channels, heads=heads, shifts=shifts, window_size=window_size)
        self.lfe = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)

    def forward(self, x, calc_attn=False):
        x = self.lfe(x) + x
        x = self.gmsa(x) + x
        return x


class Block(nn.Module):
    def __init__(self, repeats, inp_channels, out_channels, exp_ratio=2, window_size=8, heads=6):
        super(Block, self).__init__()
        block = []
        for i in range(repeats):
            if (i + 1) % 2 == 1:
                block.append(ELAB(inp_channels, out_channels, exp_ratio, 0, window_size, heads))
            else:
                block.append(ELAB(inp_channels, out_channels, exp_ratio, window_size // 2, window_size, heads))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        y = self.block(x)
        return y


# class ELAN(nn.Module):
#     def __init__(self, args, conv=common.default_conv):
#         super(ELAN, self).__init__()
#
#         n_resblocks = args.n_resblocks
#         n_feats = args.n_feats
#         kernel_size = 3
#         scale = args.scale[0]
#         act = nn.ReLU(True)
#
#         heads = 1
#         window_size = 16
#         n_resblocks = 36
#         n_feats = 180
#
#         self.scale = scale
#         self.colors = args.n_colors
#         self.window_size = window_size
#
#         self.sub_mean = common.MeanShift(args.rgb_range)
#         self.add_mean = common.MeanShift(args.rgb_range, sign=1)
#
#         # define head module
#         m_head = [conv(args.n_colors, n_feats, kernel_size)]
#
#         # define body module
#         m_body = []
#         repeats = 2
#         for i in range(n_resblocks // repeats):
#             m_body.append(Block(repeats, n_feats, n_feats, exp_ratio=2, window_size=window_size, heads=heads))
#
#         self.conv_before_tail = nn.Conv2d(n_feats, n_feats // 2, 3, 1, 1)
#         m_tail = [
#             common.Upsampler(conv, scale, n_feats // 2, act=False),
#             conv(n_feats // 2, args.n_colors, kernel_size)
#         ]
#         self.head = nn.Sequential(*m_head)
#         self.body = nn.Sequential(*m_body)
#         self.tail = nn.Sequential(*m_tail)
#
#     def forward(self, x):
#         H, W = x.shape[2:]
#         x = self.check_image_size(x)
#
#         x = self.sub_mean(x)
#         x = self.head(x)
#
#         res = self.body(x)
#         res = res + x
#
#         x = self.conv_before_tail(res)
#         x = self.tail(x)
#         x = self.add_mean(x)
#
#         return x[:, :, :H * self.scale, :W * self.scale]
#
#     def check_image_size(self, x):
#         _, _, h, w = x.size()
#         mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
#         mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
#         return x
#
#     def load_state_dict(self, state_dict, strict=True):
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name in own_state:
#                 if isinstance(param, nn.Parameter):
#                     param = param.data
#                 try:
#                     own_state[name].copy_(param)
#                 except Exception:
#                     if name.find('tail') == -1:
#                         raise RuntimeError('While copying the parameter named {}, '
#                                            'whose dimensions in the model are {} and '
#                                            'whose dimensions in the checkpoint are {}.'
#                                            .format(name, own_state[name].size(), param.size()))
#             elif strict:
#                 if name.find('tail') == -1:
#                     raise KeyError('unexpected key "{}" in state_dict'
#                                    .format(name))


if __name__ == '__main__':
    import argparse

    # x = torch.rand(1, 3, 256, 256).cpu()
    # parser = argparse.ArgumentParser(description='EDSR and MDSR')
    # parser.add_argument('--n_resblocks', type=int, default=24, help='number of residual blocks')
    # parser.add_argument('--n_feats', type=int, default=60, help='number of feature maps')
    # parser.add_argument('--scale', type=str, default='2', help='super resolution scale')
    # parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
    # parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
    # args = parser.parse_args()
    # args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    # model = ELAN(args).cpu()
    # y = model(x)
    # print(x.shape, y.shape)
