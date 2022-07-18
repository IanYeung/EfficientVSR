import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import flow_warp, resize_flow


@ARCH_REGISTRY.register()
class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch, num_feat, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        assert self.input_size == 128 or self.input_size == 256, (
            f'input size must be 128 or 256, but received {input_size}')

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

        if self.input_size == 256:
            feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
            feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


@ARCH_REGISTRY.register()
class UNetDiscriminatorWithSpectralNorm(nn.Module):
    """A U-Net discriminator with spectral normalization.
    Args:
        num_in_ch (int): Channel number of the input.
        num_feat (int, optional): Channel number of the intermediate
            features. Default: 64.
        skip_connection (bool, optional): Whether to use skip connection.
            Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):

        super().__init__()

        self.skip_connection = skip_connection

        self.conv_0 = nn.Conv2d(
            num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # downsample
        self.conv_1 = spectral_norm(
            nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv_2 = spectral_norm(
            nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv_3 = spectral_norm(
            nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        # upsample
        self.conv_4 = spectral_norm(
            nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv_5 = spectral_norm(
            nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv_6 = spectral_norm(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # final layers
        self.conv_7 = spectral_norm(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv_8 = spectral_norm(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv_9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img):
        """Forward function.
        Args:
            img (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        feat_0 = self.lrelu(self.conv_0(img))

        # downsample
        feat_1 = self.lrelu(self.conv_1(feat_0))
        feat_2 = self.lrelu(self.conv_2(feat_1))
        feat_3 = self.lrelu(self.conv_3(feat_2))

        # upsample
        feat_3 = self.upsample(feat_3)
        feat_4 = self.lrelu(self.conv_4(feat_3))
        if self.skip_connection:
            feat_4 = feat_4 + feat_2

        feat_4 = self.upsample(feat_4)
        feat_5 = self.lrelu(self.conv_5(feat_4))
        if self.skip_connection:
            feat_5 = feat_5 + feat_1

        feat_5 = self.upsample(feat_5)
        feat_6 = self.lrelu(self.conv_6(feat_5))
        if self.skip_connection:
            feat_6 = feat_6 + feat_0

        # final layers
        out = self.lrelu(self.conv_7(feat_6))
        out = self.lrelu(self.conv_8(out))

        return self.conv_9(out)


# @ARCH_REGISTRY.register()
# class SpatioTemporalDiscriminator(nn.Module):
#     """ Spatio-Temporal discriminator
#     """
#
#     def __init__(self, num_in_ch, num_feat, num_frame=3):
#         super(SpatioTemporalDiscriminator, self).__init__()
#         self.num_frame = num_frame
#         self.disc = UNetDiscriminatorWithSpectralNorm(num_in_ch=num_in_ch * num_frame, num_feat=num_feat)
#
#     def forward(self, seq):
#         b, n, c, h, w = seq.size()
#         seq = seq.permute(0, 2, 1, 3, 4).contiguous()
#         pad_size = self.num_frame // 2
#         seq = F.pad(seq, (0, 0, 0, 0, pad_size, pad_size), mode='replicate')
#         out = []
#         for i in range(n):
#             disc_input = seq[:, :, i:i+self.num_frame, :, :].view(b, c*self.num_frame, h, w)
#             out.append(self.disc(disc_input))
#         out = torch.stack(out, dim=2)
#         out = out.permute(0, 2, 1, 3, 4).contiguous()
#         return out


# @ARCH_REGISTRY.register()
# class SpatioTemporalDiscriminator(nn.Module):
#     """ Spatio-Temporal discriminator
#     """
#
#     def __init__(self, num_in_ch, num_feat, num_frame=3):
#         super(SpatioTemporalDiscriminator, self).__init__()
#         self.num_frame = num_frame
#         self.disc = UNetDiscriminatorWithSpectralNorm(num_in_ch=num_in_ch * num_frame, num_feat=num_feat)
#
#     def forward(self, seq):
#         b, n, c, h, w = seq.size()
#         seq = seq.permute(0, 2, 1, 3, 4).contiguous()
#         pad_size = self.num_frame // 2
#         seq = F.pad(seq, (0, 0, 0, 0, pad_size, pad_size), mode='replicate')
#
#         seq_list = [seq[:, :, i:i+n, :, :] for i in range(self.num_frame)]
#
#         out = []
#         for i in range(n):
#             out.append(self.disc(torch.cat([seq_list[j][:, :, i, :, :] for j in range(len(seq_list))], dim=1)))
#         out = torch.stack(out, dim=2)
#         out = out.permute(0, 2, 1, 3, 4).contiguous()
#
#         return out


@ARCH_REGISTRY.register()
class SpatioTemporalDiscriminator(nn.Module):
    """ Spatio-Temporal discriminator
    """

    def __init__(self, num_in_ch, num_feat, num_frame=3, pad_seq=False):
        super(SpatioTemporalDiscriminator, self).__init__()
        self.num_frame = num_frame
        self.pad_seq = pad_seq
        self.disc = UNetDiscriminatorWithSpectralNorm(num_in_ch=num_in_ch * num_frame, num_feat=num_feat)

    def forward(self, seq):
        if self.pad_seq:
            return self.forward_with_padding(seq)
        else:
            return self.forward_wout_padding(seq)

    def forward_with_padding(self, seq):
        b, n, c, h, w = seq.size()
        seq = seq.permute(0, 2, 1, 3, 4).contiguous()
        pad_size = self.num_frame // 2
        seq = F.pad(seq, (0, 0, 0, 0, pad_size, pad_size), mode='replicate')
        out = []
        for i in range(n):
            disc_input = seq[:, :, i:i + self.num_frame, :, :].contiguous().view(b, c * self.num_frame, h, w)
            out.append(self.disc(disc_input))
        out = torch.stack(out, dim=2)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        return out

    def forward_wout_padding(self, seq):
        b, n, c, h, w = seq.size()
        out = []
        for i in range(n - self.num_frame // 2 * 2):
            disc_input = seq[:, i:i + self.num_frame, :, :, :].contiguous().view(b, c * self.num_frame, h, w)
            out.append(self.disc(disc_input))
        out = torch.stack(out, dim=1)
        return out
