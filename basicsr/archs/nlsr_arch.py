import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.archs.arch_util import DropPath, to_2tuple, trunc_normal_


# def make_model(args, parent=False):
#     return NLSR(args)


class LayerNorm(nn.Module):
    def __init__(self, channels):
        super(LayerNorm, self).__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


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
        return (y)


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
    def __init__(self, inp_channels, out_channels, conv_type='low-training-memory'):
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


class FFN(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='gelu'):
        super(FFN, self).__init__()
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


class ATN(nn.Module):
    def __init__(self, channels, heads=8, shifts=4, window_size=8):
        super(ATN, self).__init__()
        self.channels = channels
        self.mid_channels = channels // 2
        self.heads = heads
        self.shifts = shifts
        self.window_size = window_size

        self.theta = nn.Sequential(
            ShiftConv2d(channels, channels // 2),
            nn.BatchNorm2d(channels // 2),
        )
        self.phi = nn.Sequential(
            ShiftConv2d(channels, channels // 2),
            nn.BatchNorm2d(channels // 2),
        )
        self.g = nn.Sequential(
            ShiftConv2d(channels, channels // 2),
            nn.BatchNorm2d(channels // 2)
        )
        self.project_out = ShiftConv2d(channels // 2, channels)

    def forward(self, x):
        if self.shifts > 0:
            x = F.pad(x, (self.shifts, self.shifts, self.shifts, self.shifts), 'constant', 0)
        b, c, h, w = x.shape

        theta_x = self.theta(x)
        phi_x = self.phi(x)
        g_x = self.g(x)

        theta_x = rearrange(
            theta_x, 'b (head c) (h dh) (w dw) -> (b h w) head (dh dw) c',
            head=self.heads, dh=self.window_size, dw=self.window_size
        )
        phi_x = rearrange(
            phi_x, 'b (head c) (h dh) (w dw) -> (b h w) head (dh dw) c',
            head=self.heads, dh=self.window_size, dw=self.window_size
        )
        g_x = rearrange(
            g_x, 'b (head c) (h dh) (w dw) -> (b h w) head (dh dw) c',
            head=self.heads, dh=self.window_size, dw=self.window_size
        )

        sim = (theta_x @ phi_x.transpose(-2, -1))
        sim = sim.softmax(dim=-1)

        out = (sim @ g_x)

        h_val = h // self.window_size
        w_val = w // self.window_size
        out = rearrange(
            out, '(b h w) head (dh dw) c-> b (head c) (h dh) (w dw)',
            h=h_val, w=w_val, head=self.heads, dh=self.window_size, dw=self.window_size
        )
        out = self.project_out(out)

        if self.shifts > 0:
            out = out[:, :, self.shifts:-self.shifts, self.shifts:-self.shifts]

        return out


class Transformer(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_size=9, heads=6):
        super(Transformer, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_size = window_size
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.atn = ATN(channels=inp_channels, heads=heads, shifts=shifts, window_size=window_size)
        self.ffn = FFN(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)

    def forward(self, x):
        x = self.atn(x) + x
        x = self.ffn(x) + x
        return x


class Block(nn.Module):
    def __init__(self, repeats, inp_channels, out_channels, exp_ratio=2, window_size=9, heads=6):
        super(Block, self).__init__()
        block = []
        for i in range(repeats):
            if (i + 1) % 2 == 1:
                block.append(Transformer(inp_channels, out_channels, exp_ratio, 0, window_size, heads))
            else:
                block.append(Transformer(inp_channels, out_channels, exp_ratio, window_size // 2, window_size, heads))
        self.block = nn.Sequential(*block)
        # self.conv_merge = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv_merge = ShiftConv2d(out_channels, out_channels)

    def forward(self, x):
        y = self.block(x)
        y = self.conv_merge(y) + x
        return y


# class NLSR(nn.Module):
#     def __init__(self, args, conv=common.default_conv):
#         super(NLSR, self).__init__()
#
#         n_resblocks = args.n_resblocks
#         n_feats = args.n_feats
#         kernel_size = 3
#         scale = args.scale[0]
#         act = nn.ReLU(True)
#
#         heads = 1
#         window_size = 10
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
#         repeats = 6
#         for i in range(n_resblocks // repeats):
#             m_body.append(Block(repeats, n_feats, n_feats, exp_ratio=2, window_size=window_size, heads=heads))
#
#             # define tail module
#         m_tail = [
#             nn.Conv2d(n_feats, self.colors * self.scale * self.scale, kernel_size=3, stride=1, padding=1),
#             nn.PixelShuffle(self.scale)
#         ]
#
#         self.head = nn.Sequential(*m_head)
#         self.body = nn.Sequential(*m_body)
#         self.after_body = conv(n_feats, n_feats, kernel_size)
#         self.norm = LayerNorm(n_feats)
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
#         res = self.norm(res)
#         res = self.after_body(res) + x
#
#         x = self.tail(res)
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

    # import argparse
    # x = torch.rand(1, 3, 480, 480)
    # parser = argparse.ArgumentParser(description='EDSR and MDSR')
    # parser.add_argument('--n_resblocks', type=int, default=24, help='number of residual blocks')
    # parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    # parser.add_argument('--scale', type=str, default='2', help='super resolution scale')
    # parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
    # parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
    # args = parser.parse_args()
    # args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    # model = CVIR(args)
    # y = model(x)
    # print(x.shape, y.shape)

    x = torch.rand(1, 32, 64, 64)

    shift_conv = ShiftConv2d(inp_channels=32, out_channels=32)
    y = shift_conv(x)
    print(y.shape)

    atn = ATN(channels=32, heads=8)
    y = atn(x)
    print(y.shape)
