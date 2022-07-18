import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import make_layer, flow_warp, FirstOrderDeformableAlignment
from .arch_util import ResidualBlockNoBN


class PyramidAttentionAlignV1(nn.Module):
    """Pyramid Attention Alignment.

    Args:
        num_feat (int): Channel number of intermediate features. Default: 64.
    """

    def __init__(self, num_feat=64):
        super(PyramidAttentionAlignV1, self).__init__()

        self.feature_extract = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # 1/1
        self.conv_l1_1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l1_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l1_o = nn.Conv2d(num_feat, 1, 3, 1, 1)  # attention output

        # 1/2
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l2_o = nn.Conv2d(num_feat, 1, 3, 1, 1)  # attention output

        # 1/4
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_o = nn.Conv2d(num_feat, 1, 3, 1, 1)  # attention output

        # 1/8
        self.conv_l4_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l4_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l4_o = nn.Conv2d(num_feat, 1, 3, 1, 1)  # attention output

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea, ref_fea):
        # construct feature pyramid
        inp_fea = self.lrelu(self.feature_extract(torch.cat([nbr_fea, ref_fea], dim=1)))
        # L1
        feat_l1 = self.lrelu(self.conv_l1_1(inp_fea))
        feat_l1 = self.lrelu(self.conv_l1_2(feat_l1))
        attn_l1 = self.conv_l1_o(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        attn_l2 = self.conv_l2_o(feat_l2)
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
        attn_l3 = self.conv_l3_o(feat_l3)
        # L4
        feat_l4 = self.lrelu(self.conv_l4_1(feat_l3))
        feat_l4 = self.lrelu(self.conv_l4_2(feat_l4))
        attn_l4 = self.conv_l4_o(feat_l4)

        out_fea_l1 = attn_l1 * feat_l1
        out_fea_l2 = F.interpolate(attn_l2, scale_factor=2, mode='bilinear', align_corners=False) * feat_l1
        out_fea_l3 = F.interpolate(attn_l3, scale_factor=4, mode='bilinear', align_corners=False) * feat_l1
        out_fea_l4 = F.interpolate(attn_l4, scale_factor=8, mode='bilinear', align_corners=False) * feat_l1

        out_fea = out_fea_l1 + out_fea_l2 + out_fea_l3 + out_fea_l4

        return out_fea


class PyramidAttentionAlignV2(nn.Module):
    """Pyramid Attention Alignment.

    Args:
        num_feat (int): Channel number of intermediate features. Default: 64.
    """

    def __init__(self, num_feat=64):
        super(PyramidAttentionAlignV2, self).__init__()

        self.feature_extract = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # 1/1
        self.conv_l1_1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l1_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l1_o = nn.Conv2d(num_feat, 1, 3, 1, 1)  # attention output

        # 1/2
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l2_o = nn.Conv2d(num_feat, 1, 3, 1, 1)  # attention output

        # 1/4
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_o = nn.Conv2d(num_feat, 1, 3, 1, 1)  # attention output

        # 1/8
        self.conv_l4_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l4_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l4_o = nn.Conv2d(num_feat, 1, 3, 1, 1)  # attention output

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea, ref_fea):
        # construct feature pyramid
        inp_fea = self.lrelu(self.feature_extract(torch.cat([nbr_fea, ref_fea], dim=1)))
        # L1
        feat_l1 = self.lrelu(self.conv_l1_1(inp_fea))
        feat_l1 = self.lrelu(self.conv_l1_2(feat_l1))
        attn_l1 = self.conv_l1_o(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        attn_l2 = self.conv_l2_o(feat_l2)
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
        attn_l3 = self.conv_l3_o(feat_l3)
        # L4
        feat_l4 = self.lrelu(self.conv_l4_1(feat_l3))
        feat_l4 = self.lrelu(self.conv_l4_2(feat_l4))
        attn_l4 = self.conv_l4_o(feat_l4)

        attn_weight = F.interpolate(attn_l4, scale_factor=8, mode='bilinear', align_corners=False) + \
                      F.interpolate(attn_l3, scale_factor=4, mode='bilinear', align_corners=False) + \
                      F.interpolate(attn_l2, scale_factor=2, mode='bilinear', align_corners=False) + \
                      attn_l1

        out_fea = attn_weight * nbr_fea

        return out_fea


@ARCH_REGISTRY.register()
class PyramidAttentionAlignVSR(nn.Module):
    """PyramidAttentionAlignVSR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: Middle of input frames.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None):
        super(PyramidAttentionAlignVSR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        # alignment
        self.pyr_att_align = PyramidAttentionAlignV2(num_feat=num_feat)

        # fusion
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

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        feat_l1 = feat_l1.view(b, t, -1, h, w)

        # alignment
        aligned_feat = []
        for i in range(t):
            cen = feat_l1[:, self.center_frame_idx, :, :, :].contiguous()
            nbr = feat_l1[:, i, :, :, :].contiguous()
            aligned_feat.append(self.pyr_att_align(nbr, cen))
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