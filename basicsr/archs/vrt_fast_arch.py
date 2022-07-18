import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from basicsr.archs.arch_util import make_layer, ResidualBlockNoBN, ResidualBlocksWithInputConv
from basicsr.archs.vrt_arch import TMSAG, DCNv2PackFlowGuided, Mlp_GEGLU, Upsample, SpyNet, flow_warp
from basicsr.utils.registry import ARCH_REGISTRY


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


class Stage(nn.Module):
    """Residual Temporal Mutual Self Attention Group and Parallel Warping.

    Args:
        in_dim (int): Number of input channels.
        dim (int): Number of channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        reshape (str): Downscale (down), upscale (up) or keep the size (none).
        max_residue_magnitude (float): Maximum magnitude of the residual of optical flow.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 in_dim,
                 dim,
                 # input_resolution,
                 # depth,
                 # num_heads,
                 # window_size,
                 # mul_attn_ratio=0.75,
                 # mlp_ratio=2.,
                 # qkv_bias=True,
                 # qk_scale=None,
                 # drop_path=0.,
                 # norm_layer=nn.LayerNorm,
                 pa_frames=2,
                 deformable_groups=16,
                 reshape='none',
                 max_residue_magnitude=10,
                 ):
        super(Stage, self).__init__()
        self.pa_frames = pa_frames

        # reshape the tensor
        if reshape == 'none':
            self.reshape = nn.Sequential(
                Rearrange('n c d h w -> n d h w c'),
                nn.LayerNorm(dim), nn.Linear(in_dim, dim)
            )
        elif reshape == 'down':
            self.reshape = nn.Sequential(
                Rearrange('n c d (h neih) (w neiw) -> n d h w (neiw neih c)', neih=2, neiw=2),
                nn.LayerNorm(4 * in_dim), nn.Linear(4 * in_dim, dim)
            )
        elif reshape == 'up':
            self.reshape = nn.Sequential(
                Rearrange('n (neiw neih c) d h w -> n d (h neih) (w neiw) c', neih=2, neiw=2),
                nn.LayerNorm(in_dim // 4), nn.Linear(in_dim // 4, dim)
            )

        self.linear = nn.Linear(dim, dim)

        # parallel warping
        self.pa_deform = DCNv2PackFlowGuided(dim, dim, 3, padding=1, deformable_groups=deformable_groups,
                                             max_residue_magnitude=max_residue_magnitude, pa_frames=pa_frames)
        self.pa_fuse = Mlp_GEGLU(dim * (1 + 2), dim * (1 + 2), dim)

    def forward(self, x, flows_backward, flows_forward):
        """

        :param x: (B, D, C, H, W)
        :param flows_backward:
        :param flows_forward:
        :return:
        """
        x = self.reshape(x)
        x = self.linear(x) + x
        x = rearrange(x, 'b d h w c -> b d c h w')

        x_backward, x_forward = getattr(self, f'get_aligned_feature_{self.pa_frames}frames')(x, flows_backward, flows_forward)
        x = self.pa_fuse(torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)

        return x

    def get_aligned_feature_2frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 2 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[0][:, i - 1, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            x_backward.insert(0, self.pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[0][:, i, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_forward.append(self.pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_4frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 4 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n, 1, -1):
            x_i = x[:, i - 1, ...]
            flow1 = flows_backward[0][:, i - 2, ...]
            if i == n:
                x_ii = torch.zeros_like(x[:, n - 2, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, n - 3, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_backward[1][:, i - 2, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
            x_backward.insert(0,
                self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i - 2, ...], [flow1, flow2]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(-1, n - 2):
            x_i = x[:, i + 1, ...]
            flow1 = flows_forward[0][:, i + 1, ...]
            if i == -1:
                x_ii = torch.zeros_like(x[:, 1, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_forward[1][:, i, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
            x_forward.append(
                self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i + 2, ...], [flow1, flow2]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_6frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 6 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n + 1, 2, -1):
            x_i = x[:, i - 2, ...]
            flow1 = flows_backward[0][:, i - 3, ...]
            if i == n + 1:
                x_ii = torch.zeros_like(x[:, -1, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, -1, ...])
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            elif i == n:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = x[:, i, ...]
                flow3 = flows_backward[2][:, i - 3, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i+3 aligned towards i
            x_backward.insert(0,
                              self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                             x[:, i - 3, ...], [flow1, flow2, flow3]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow1 = flows_forward[0][:, i, ...]
            if i == 0:
                x_ii = torch.zeros_like(x[:, 0, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            elif i == 1:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = x[:, i - 2, ...]
                flow3 = flows_forward[2][:, i - 2, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i-3 aligned towards i
            x_forward.append(self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                            x[:, i + 1, ...], [flow1, flow2, flow3]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]


@ARCH_REGISTRY.register()
class FastVRT(nn.Module):
    """ Video Restoration Transformer (VRT).
        A PyTorch impl of : `VRT: A Video Restoration Transformer`  -
          https://arxiv.org/pdf/2201.00000

    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        img_size (int | tuple(int)): Size of input image. Default: [6, 64, 64].
        window_size (int | tuple(int)): Window size. Default: (6,8,8).
        depths (list[int]): Depths of each Transformer stage.
        indep_reconsts (list[int]): Layers that extract features of different frames independently.
        embed_dims (list[int]): Number of linear projection output channels.
        num_heads (list[int]): Number of attention head of each stage.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        spynet_path (str): Pretrained SpyNet model path.
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        recal_all_flows (bool): If True, derive (t,t+2) and (t,t+3) flows from (t,t+1). Default: False.
        nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
        no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
        no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
    """

    def __init__(self,
                 upscale=4,
                 in_chans=3,
                 num_feat=64,
                 num_recon_block=40,
                 norm_layer=nn.LayerNorm,
                 spynet_path=None,
                 pa_frames=2,
                 deformable_groups=16,
                 recal_all_flows=False,
                 ):
        super().__init__()
        self.in_chans = in_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.recal_all_flows = recal_all_flows

        # conv_first
        self.conv_first = nn.Conv3d(in_chans, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # main body
        self.spynet = SpyNet(spynet_path, [3, 4, 5])

        reshapes = ['none', 'down', 'down', 'up', 'up']
        scales = [1, 2, 4, 2, 1]

        # stage 1- 7
        for i in range(5):
            setattr(self, f'stage{i + 1}',
                    Stage(
                        in_dim=num_feat,
                        dim=num_feat,
                        pa_frames=pa_frames,
                        deformable_groups=deformable_groups,
                        reshape=reshapes[i],
                        max_residue_magnitude=10 / scales[i],
                        )
                    )

        self.norm = norm_layer(num_feat)
        self.conv_after_body = nn.Linear(num_feat, num_feat)

        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_recon_block, num_feat=num_feat)

        # for video sr
        self.conv_before_upsample = nn.Sequential(
            nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv3d(num_feat, in_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        # x: (N, D, C, H, W)

        x_lq = x.clone()

        # calculate flows
        flows_backward, flows_forward = self.get_flows(x)

        # main network
        x = rearrange(x, 'b n c h w -> b c n h w')
        x = self.conv_first(x)
        x = x + self.conv_after_body(self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
        x = self.conv_last(self.upsample(self.conv_before_upsample(x)))
        x = rearrange(x, 'b c n h w -> b n c h w')

        _, _, C, H, W = x.shape
        return x + torch.nn.functional.interpolate(x_lq, size=(C, H, W), mode='trilinear', align_corners=False)

    def get_flows(self, x):
        ''' Get flows for 2 frames, 4 frames or 6 frames.'''

        if self.pa_frames == 2:
            flows_backward, flows_forward = self.get_flow_2frames(x)
        elif self.pa_frames == 4:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames
            flows_forward = flows_forward_2frames + flows_forward_4frames
        elif self.pa_frames == 6:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward_6frames, flows_forward_6frames = self.get_flow_6frames(flows_forward_2frames, flows_backward_2frames, flows_forward_4frames, flows_backward_4frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames + flows_backward_6frames
            flows_forward = flows_forward_2frames + flows_forward_4frames + flows_forward_6frames

        return flows_backward, flows_forward

    def get_flow_2frames(self, x):
        '''Get flow between frames t and t+1 from x.'''

        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        # backward
        flows_backward = self.spynet(x_1, x_2)
        flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                          zip(flows_backward, range(4))]

        # forward
        flows_forward = self.spynet(x_2, x_1)
        flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                         zip(flows_forward, range(4))]

        return flows_backward, flows_forward

    def get_flow_4frames(self, flows_forward, flows_backward):
        '''Get flow between t and t+2 from (t,t+1) and (t+1,t+2).'''

        # backward
        d = flows_forward[0].shape[1]
        flows_backward2 = []
        for flows in flows_backward:
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows[:, i - 1, :, :, :]  # flow from i+1 to i
                flow_n2 = flows[:, i, :, :, :]  # flow from i+2 to i+1
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+2 to i
            flows_backward2.append(torch.stack(flow_list, 1))

        # forward
        flows_forward2 = []
        for flows in flows_forward:
            flow_list = []
            for i in range(1, d):
                flow_n1 = flows[:, i, :, :, :]  # flow from i-1 to i
                flow_n2 = flows[:, i - 1, :, :, :]  # flow from i-2 to i-1
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-2 to i
            flows_forward2.append(torch.stack(flow_list, 1))

        return flows_backward2, flows_forward2

    def get_flow_6frames(self, flows_forward, flows_backward, flows_forward2, flows_backward2):
        '''Get flow between t and t+3 from (t,t+2) and (t+2,t+3).'''

        # backward
        d = flows_forward2[0].shape[1]
        flows_backward3 = []
        for flows, flows2 in zip(flows_backward, flows_backward2):
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i+2 to i
                flow_n2 = flows[:, i + 1, :, :, :]  # flow from i+3 to i+2
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+3 to i
            flows_backward3.append(torch.stack(flow_list, 1))

        # forward
        flows_forward3 = []
        for flows, flows2 in zip(flows_forward, flows_forward2):
            flow_list = []
            for i in range(2, d + 1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i-2 to i
                flow_n2 = flows[:, i - 2, :, :, :]  # flow from i-3 to i-2
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-3 to i
            flows_forward3.append(torch.stack(flow_list, 1))

        return flows_backward3, flows_forward3

    def forward_features(self, x, flows_backward, flows_forward):
        '''Main network for feature extraction.'''

        x1 = self.stage1(x, flows_backward[0::3], flows_forward[0::3])
        x2 = self.stage2(x1, flows_backward[1::3], flows_forward[1::3])
        x3 = self.stage3(x2, flows_backward[2::3], flows_forward[2::3])
        x = self.stage4(x3, flows_backward[1::3], flows_forward[1::3])
        x = self.stage5(x + x2, flows_backward[0::3], flows_forward[0::3])
        x = x + x1

        b, c, d, h, w = x.shape
        x = rearrange(x, 'b c d h w -> (b d) c h w')
        x = self.reconstruction(x)
        x = rearrange(x, '(b d) c h w -> b c d h w', b=b, d=d)

        return x


if __name__ == '__main__':

    device = torch.device('cuda')

    # import numpy as np
    #
    # model = SelfAttBlock(64 * (2 ** 2), 64 * (2 ** 2)).to(device)
    # # model = SelfAttBlock(64, 64).to(device)
    #
    # # INIT LOGGERS
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 10
    # timings = np.zeros((repetitions, 1))
    #
    # # GPU-WARM-UP
    # # for _ in range(2):
    # #     dummy_input = torch.randn(1, 64, 256, 256).to(device)
    # #     _ = model(dummy_input)
    #
    # # MEASURE PERFORMANCE
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         dummy_input = torch.randn(1, 64, 256, 256).to(device)
    #         starter.record()
    #
    #         _ = F.pixel_shuffle(model(F.pixel_unshuffle(dummy_input, downscale_factor=2)), upscale_factor=2)
    #         # _ = model(dummy_input)
    #
    #         ender.record()
    #         # WAIT FOR GPU SYNC
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time
    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # print(mean_syn)

    # attention = SelfAttBlock(64, 64)
    # inp = torch.rand(1, 64, 256, 256)
    # out = attention(inp)
    # print(out.shape)
    # cross_blk = CrossAttBlock(64, 64, exp_ratio=1, shifts=0, window_size=4, heads=4, norm=True)
    # inp1 = torch.rand(1, 64, 256, 256)
    # inp2 = torch.rand(1, 64, 256, 256)
    # out = cross_blk(inp1, inp2)
    # print(out.shape)

    class Attention(nn.Module):

        def __init__(self, channels, heads=8, shifts=4, window_size=8):
            super(Attention, self).__init__()
            self.channels = channels
            self.heads = heads
            self.shifts = shifts
            self.window_size = window_size

            self.project_inp = nn.Linear(channels, channels * 3)
            self.project_out = nn.Linear(channels, channels)
            self.scale = (channels // heads) ** -0.5

            # self.norm = LayerNorm(3 * channels)

        def forward(self, x):
            b, n, h, w, c = x.shape

            if self.shifts > 0:
                x = torch.roll(x, shifts=(-self.shifts, -self.shifts), dims=(2, 3))

            x = self.project_inp(x)

            q, k, v = rearrange(
                x, 'b n (h dh) (w dw) (qkv head c) -> qkv (b n h w) head (dh dw) c',
                qkv=3, head=self.heads, dh=self.window_size, dw=self.window_size
            )

            atn = (q @ k.transpose(-2, -1)) * self.scale
            atn = atn.softmax(dim=-1)

            out = (atn @ v)
            out = rearrange(
                out, '(b n h w) head (dh dw) c -> b n (h dh) (w dw) (head c)',
                n=n,
                h=h // self.window_size,
                w=w // self.window_size,
                head=self.heads,
                dh=self.window_size,
                dw=self.window_size
            )
            out = self.project_out(out)

            if self.shifts > 0:
                out = torch.roll(out, shifts=(self.shifts, self.shifts), dims=(2, 3))

            return out

    # attention = Attention(64)
    # inp = torch.rand(1, 5, 128, 128, 64)
    # out = attention(inp)
    # print(out.shape)

    # stage = Stage(in_dim=64, dim=64).to(device)
    # inp = torch.rand(1, 5, 128, 128, 64).to(device)
    # flow_1 = torch.rand(1, 4, 2, 128, 128).to(device)
    # flow_2 = torch.rand(1, 4, 2, 128, 128).to(device)
    # out = stage(inp, [flow_1], [flow_2])
    # print(out.shape)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    inp = torch.rand(1, 5, 3, 64, 64)
    net = FastVRT()
    out = net(inp)
    print(out.shape)

    print(count_parameters(net) / 10 ** 6)

    b, n, c, h, w = 1, 50, 3, 144, 180

    model = FastVRT()
    model.eval()
    model.to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 10
    timings = np.zeros((repetitions, 1))

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            print('Repetition: {:02d}.'.format(rep))
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
