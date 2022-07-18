import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import flow_warp, make_layer, ResidualBlockNoBN, CoTResidualBlock, CoTResBottleneck
from .spynet_arch import SpyNet
from .restormer_arch import OverlapPatchEmbed, TransformerBlock


@ARCH_REGISTRY.register()
class FlowImgVSR(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 spynet_path=None):
        super(FlowImgVSR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.conv_first = nn.Conv2d(num_in_ch * num_frame, num_feat, 3, 1, 1)

        self.spynet = SpyNet(load_path=spynet_path)

        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()

        assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # alignment
        aligned_imgs = []
        for i in range(t):
            flow = self.spynet(x_center, x[:, i, :, :, :].contiguous())
            nbr = x[:, i, :, :, :].contiguous()
            aligned_imgs.append(flow_warp(nbr, flow.permute(0, 2, 3, 1)))

        aligned_imgs = torch.stack(aligned_imgs, dim=1)  # (b, t, c, h, w)

        aligned_imgs = aligned_imgs.view(b, -1, h, w)

        feat = self.conv_first(aligned_imgs)
        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        return out


@ARCH_REGISTRY.register()
class FlowFeaVSR(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 spynet_path=None):
        super(FlowFeaVSR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        # extract features for each frame
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        self.spynet = SpyNet(load_path=spynet_path)
        self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()

        assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        feat_l1 = feat_l1.view(b, t, -1, h, w)

        # alignment
        aligned_feat = []
        for i in range(t):
            flow = self.spynet(x_center, x[:, i, :, :, :].contiguous())
            nbr = feat_l1[:, i, :, :, :].contiguous()
            aligned_feat.append(flow_warp(nbr, flow.permute(0, 2, 3, 1)))

        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        return out


@ARCH_REGISTRY.register()
class FlowFeaCoTVSR(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 spynet_path=None):
        super(FlowFeaCoTVSR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        # extract features for each frame
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        self.spynet = SpyNet(load_path=spynet_path)
        self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = make_layer(CoTResidualBlock, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()

        assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        feat_l1 = feat_l1.view(b, t, -1, h, w)

        # alignment
        aligned_feat = []
        for i in range(t):
            flow = self.spynet(x_center, x[:, i, :, :, :].contiguous())
            nbr = feat_l1[:, i, :, :, :].contiguous()
            aligned_feat.append(flow_warp(nbr, flow.permute(0, 2, 3, 1)))

        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        return out


@ARCH_REGISTRY.register()
class FlowImgChannelTransformerVSR(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 spynet_path=None):
        super(FlowImgChannelTransformerVSR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        # flownet
        self.spynet = SpyNet(load_path=spynet_path)

        self.overlap_patch_embed = OverlapPatchEmbed(in_c=num_in_ch * num_frame, embed_dim=num_feat, bias=False)

        # reconstruction
        self.reconstruction = make_layer(TransformerBlock, num_reconstruct_block, dim=64, num_heads=4,
                                         ffn_expansion_factor=2, bias=True, LayerNorm_type='WithBias')

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()

        assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # alignment
        aligned_imgs = []
        for i in range(t):
            flow = self.spynet(x_center, x[:, i, :, :, :].contiguous())
            nbr = x[:, i, :, :, :].contiguous()
            aligned_imgs.append(flow_warp(nbr, flow.permute(0, 2, 3, 1)))

        aligned_imgs = torch.stack(aligned_imgs, dim=1)  # (b, t, c, h, w)

        aligned_imgs = aligned_imgs.view(b, -1, h, w)

        feat = self.overlap_patch_embed(aligned_imgs)
        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        return out


@ARCH_REGISTRY.register()
class FlowFeaChannelTransformerVSR(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 spynet_path=None):
        super(FlowFeaChannelTransformerVSR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        # extract features for each frame
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        # flownet
        self.spynet = SpyNet(load_path=spynet_path)

        self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.overlap_patch_embed = OverlapPatchEmbed(in_c=num_feat, embed_dim=num_feat, bias=False)
        self.reconstruction = make_layer(TransformerBlock, num_reconstruct_block, dim=64, num_heads=4,
                                         ffn_expansion_factor=2, bias=True, LayerNorm_type='WithBias')

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()

        assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        feat_l1 = feat_l1.view(b, t, -1, h, w)

        # alignment
        aligned_feat = []
        for i in range(t):
            flow = self.spynet(x_center, x[:, i, :, :, :].contiguous())
            nbr = feat_l1[:, i, :, :, :].contiguous()
            aligned_feat.append(flow_warp(nbr, flow.permute(0, 2, 3, 1)))

        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        feat = self.overlap_patch_embed(feat)
        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        return out