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
    return CVIR(args)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


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


class FFN(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='gelu'):
        super(FFN, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type

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
        y = self.conv1x1_0(x)
        y = self.act(y)
        y = self.conv1x1_1(y)
        return y


class ATN(nn.Module):
    def __init__(self, channels, heads=8, shifts=4, window_size=8):
        super(ATN, self).__init__()
        self.channels = channels
        self.heads = heads
        self.shifts = shifts
        self.window_size = window_size

        self.project_inp = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = (channels // heads) ** -0.5

        # self.norm = LayerNorm(3 * channels)

    def forward(self, x):
        b, c, h, w = x.shape

        if self.shifts > 0:
            x = torch.roll(x, shifts=(-self.shifts, -self.shifts), dims=(2, 3))

        x = self.project_inp(x)

        q, k, v = rearrange(
            x, 'b (qkv head c) (h dh) (w dw) -> qkv (b h w) head (dh dw) c',
            qkv=3, head=self.heads, dh=self.window_size, dw=self.window_size
        )

        atn = (q @ k.transpose(-2, -1)) * self.scale
        atn = atn.softmax(dim=-1)

        out = (atn @ v)
        out = rearrange(
            out, '(b h w) head (dh dw) c-> b (head c) (h dh) (w dw)',
            h=h // self.window_size, w=w // self.window_size, head=self.heads, dh=self.window_size, dw=self.window_size
        )
        out = self.project_out(out)

        if self.shifts > 0:
            out = torch.roll(out, shifts=(self.shifts, self.shifts), dims=(2, 3))

        return out


class Transformer(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_size=8, heads=8, norm=True):
        super(Transformer, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_size = window_size
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.atn = ATN(channels=inp_channels, heads=heads, shifts=shifts, window_size=window_size)
        self.ffn = FFN(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)

        if norm:
            self.norm = LayerNorm(inp_channels)
        else:
            self.norm = Identity()

    def forward(self, x):
        x = self.atn(self.norm(x)) + x
        x = self.ffn(self.norm(x)) + x
        return x


class SwinBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, repeats=6, exp_ratio=2, window_size=8, heads=8, norm=True):
        super(SwinBlock, self).__init__()
        block = []
        for i in range(repeats):
            if (i + 1) % 2 == 1:
                block.append(
                    Transformer(inp_channels, out_channels, exp_ratio, 0, window_size, heads, norm=norm)
                )
            else:
                block.append(
                    Transformer(inp_channels, out_channels, exp_ratio, window_size // 2, window_size, heads, norm=norm)
                )
        self.block = nn.Sequential(*block)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        y = self.block(x)
        y = self.conv3x3(y) + x
        return y


class CrossATN(nn.Module):
    def __init__(self, channels, heads=8, shifts=4, window_size=8):
        super(CrossATN, self).__init__()
        self.channels = channels
        self.heads = heads
        self.shifts = shifts
        self.window_size = window_size

        self.project_q_ = nn.Conv2d(channels, channels * 1, kernel_size=1)
        self.project_kv = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = (channels // heads) ** -0.5

        # self.norm = LayerNorm(3 * channels)

    def forward(self, x, y):
        b, c, h, w = x.shape

        if self.shifts > 0:
            x = torch.roll(x, shifts=(-self.shifts, -self.shifts), dims=(2, 3))
            y = torch.roll(y, shifts=(-self.shifts, -self.shifts), dims=(2, 3))

        x = self.project_q_(x)
        y = self.project_kv(y)

        q = rearrange(
            x, 'b (qu head c) (h dh) (w dw) -> qu (b h w) head (dh dw) c',
            qu=1, head=self.heads, dh=self.window_size, dw=self.window_size
        ).squeeze(0)

        k, v = rearrange(
            y, 'b (kv head c) (h dh) (w dw) -> kv (b h w) head (dh dw) c',
            kv=2, head=self.heads, dh=self.window_size, dw=self.window_size
        )

        atn = (q @ k.transpose(-2, -1)) * self.scale
        atn = atn.softmax(dim=-1)

        out = (atn @ v)
        out = rearrange(
            out, '(b h w) head (dh dw) c-> b (head c) (h dh) (w dw)',
            h=h // self.window_size, w=w // self.window_size, head=self.heads, dh=self.window_size, dw=self.window_size
        )
        out = self.project_out(out)

        if self.shifts > 0:
            out = torch.roll(out, shifts=(self.shifts, self.shifts), dims=(2, 3))

        return out


class SelfAttBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_size=8, heads=8, norm=True):
        super(SelfAttBlock, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_size = window_size
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.atn = ATN(channels=inp_channels, heads=heads, shifts=shifts, window_size=window_size)
        self.ffn = FFN(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)

        if norm:
            self.norm = LayerNorm(inp_channels)
        else:
            self.norm = Identity()

    def forward(self, x):
        x = self.atn(self.norm(x)) + x
        x = self.ffn(self.norm(x)) + x
        return x


class CrossAttBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_size=8, heads=8, norm=True):
        super(CrossAttBlock, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_size = window_size
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.atn = CrossATN(channels=inp_channels, heads=heads, shifts=shifts, window_size=window_size)
        self.ffn = FFN(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)

        if norm:
            self.norm = LayerNorm(inp_channels)
        else:
            self.norm = Identity()

    def forward(self, x, y):
        x = self.atn(self.norm(x), self.norm(y)) + x
        x = self.ffn(self.norm(x)) + x
        return x


# class CVIR(nn.Module):
#     def __init__(self, args, conv=common.default_conv):
#         super(CVIR, self).__init__()
#
#         n_resblocks = args.n_resblocks
#         n_feats = args.n_feats
#         kernel_size = 3
#         scale = args.scale[0]
#         act = nn.ReLU(True)
#
#         heads = 6
#         window_size = 8
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
#             m_body.append(SwinBlock(repeats, n_feats, n_feats, exp_ratio=2, window_size=window_size, heads=heads))
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
#         res = self.after_body(self.norm(res)) + x
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

    device = torch.device('cuda')
    # attention = SelfAttBlock(64 * (2 ** 2), 64 * (2 ** 2)).to(device)
    # inp = torch.randn(1, 64, 256, 256).to(device)
    # inp_down = F.pixel_unshuffle(inp, downscale_factor=2)
    # out_down = attention(inp_down)
    # out = F.pixel_shuffle(out_down, upscale_factor=2)
    # print(out.shape)
    # attention = SelfAttBlock(64, 64).to(device)
    # inp = torch.randn(1, 64, 256, 256).to(device)
    # out = attention(inp)
    # print(out.shape)

    import numpy as np

    model = SelfAttBlock(64 * (2 ** 2), 64 * (2 ** 2)).to(device)
    # model = SelfAttBlock(64, 64).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 10
    timings = np.zeros((repetitions, 1))

    # GPU-WARM-UP
    # for _ in range(2):
    #     dummy_input = torch.randn(1, 64, 256, 256).to(device)
    #     _ = model(dummy_input)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            dummy_input = torch.randn(1, 64, 256, 256).to(device)
            starter.record()

            _ = F.pixel_shuffle(model(F.pixel_unshuffle(dummy_input, downscale_factor=2)), upscale_factor=2)
            # _ = model(dummy_input)

            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)

    # attention = SelfAttBlock(64, 64)
    # inp = torch.rand(1, 64, 256, 256)
    # out = attention(inp)
    # print(out.shape)
    # cross_blk = CrossAttBlock(64, 64, exp_ratio=1, shifts=0, window_size=4, heads=4, norm=True)
    # inp1 = torch.rand(1, 64, 256, 256)
    # inp2 = torch.rand(1, 64, 256, 256)
    # out = cross_blk(inp1, inp2)
    # print(out.shape)

    # swin_block = SwinBlock(repeats=8, inp_channels=64, out_channels=64, heads=8)
    # x = torch.rand(1, 64, 256, 256)
    # y = swin_block(x)
    # print(y.shape)

    # import argparse
    #
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
