import torch
from torch import nn as nn
from torch.nn import functional as F

# from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ConvReLU, ResidualBlockNoBN, IterativeAlignmentUnit, make_layer


class FeatureExtractionModule(nn.Module):
    """Feature Extraction Module.

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super(FeatureExtractionModule, self).__init__()
        self.hr_in = hr_in

        if self.hr_in:
            # downsample x4 by stride conv
            self.stride_2_conv_1 = ConvReLU(num_in_ch, num_feat, 3, 2, 1, has_relu=True)
            self.stride_2_conv_2 = ConvReLU(num_feat, num_feat, 3, 2, 1, has_relu=True)
        else:
            self.stride_1_conv_1 = ConvReLU(num_in_ch, num_feat, 3, 1, 1, has_relu=True)
            self.stride_1_conv_2 = ConvReLU(num_feat, num_feat, 3, 1, 1, has_relu=True)

        self.down_conv_1 = ConvReLU(num_feat, num_feat, 3, 2, 1, has_relu=True)
        self.down_conv_2 = ConvReLU(num_feat, num_feat, 3, 2, 1, has_relu=True)

        self.fusion_conv = ConvReLU(num_feat * 3, num_feat, 3, 1, 1, has_relu=True)

        self.residual_block = ResidualBlockNoBN(num_feat=num_feat)

        self.bicubic_upsample_x2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.bicubic_upsample_x4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):

        if self.hr_in:
            feat = self.stride_2_conv_2(self.stride_2_conv_1(x))
        else:
            feat = self.stride_1_conv_2(self.stride_1_conv_1(x))
        feat = self.residual_block(feat)
        down_x1_feat = feat
        down_x2_feat = self.down_conv_1(down_x1_feat)
        down_x4_feat = self.down_conv_2(down_x2_feat)
        feat1 = down_x1_feat
        feat2 = self.bicubic_upsample_x2(down_x2_feat)
        feat4 = self.bicubic_upsample_x4(down_x4_feat)
        feat = torch.cat([feat1, feat2, feat4], dim=1)

        return self.fusion_conv(feat)


# class LocalCorr(torch.nn.Module):
#     def __init__(self, nf, nbr_size=3, alpha=-1.0):
#         super(LocalCorr, self).__init__()
#         self.nf = nf
#         self.nbr_size = nbr_size
#         self.alpha = alpha
#
#     def forward(self, nbr_list, ref):
#         mean = torch.stack(nbr_list, 1).mean(1).detach().clone()
#         # print(mean.shape)
#         b, c, h, w = ref.size()
#         ref_clone = ref.detach().clone()
#         ref_flat = ref_clone.view(b, c, -1, h * w).permute(0, 3, 2, 1).contiguous().view(b * h * w, -1, c)
#         ref_flat = F.normalize(ref_flat, p=2, dim=-1)
#         pad = self.nbr_size // 2
#         afea_list = []
#         for i in range(len(nbr_list)):
#             nbr = nbr_list[i]
#             weight_diff = (nbr - mean) ** 2
#             weight_diff = torch.exp(self.alpha * weight_diff)
#
#             nbr_pad = F.pad(nbr, (pad, pad, pad, pad), mode='reflect')
#             nbr = F.unfold(nbr_pad, kernel_size=self.nbr_size).view(b, c, -1, h * w)
#             nbr = F.normalize(nbr, p=2, dim=1)
#             nbr = nbr.permute(0, 3, 1, 2).contiguous().view(b * h * w, c, -1)
#             d = torch.matmul(ref_flat, nbr).squeeze(1)
#             weight_temporal = F.softmax(d, -1)
#             agg_fea = torch.einsum('bc,bnc->bn', weight_temporal, nbr).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
#
#             agg_fea = agg_fea * weight_diff
#
#             afea_list.append(agg_fea)
#         al_fea = torch.stack(afea_list + [ref], 1)
#         return al_fea


class LocalCorr(torch.nn.Module):

    def __init__(self, nf, nbr_size=3):
        super(LocalCorr, self).__init__()
        self.nf = nf
        self.nbr_size = nbr_size

    def forward(self, nbr, ref):

        b, c, h, w = ref.size()
        ref_clone = ref.detach().clone()
        ref_flat = ref_clone.view(b, c, -1, h * w).permute(0, 3, 2, 1).contiguous().view(b * h * w, -1, c)
        ref_flat = F.normalize(ref_flat, p=2, dim=-1)
        pad = self.nbr_size // 2

        nbr_pad = F.pad(nbr, (pad, pad, pad, pad), mode='reflect')
        nbr = F.unfold(nbr_pad, kernel_size=self.nbr_size).view(b, c, -1, h * w)
        nbr = F.normalize(nbr, p=2, dim=1)
        nbr = nbr.permute(0, 3, 1, 2).contiguous().view(b * h * w, c, -1)
        d = torch.matmul(ref_flat, nbr).squeeze(1)
        weight_temporal = F.softmax(d, -1)
        agg_fea = torch.einsum('bc,bnc->bn', weight_temporal, nbr).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

        return agg_fea


if __name__ == '__main__':

    device = torch.device('cuda')

    # fea_module = FeatureExtractionModule(num_in_ch=3, num_feat=64, hr_in=True).to(device)
    # inp = torch.randn(1, 3, 128, 128).to(device)
    # out = fea_module(inp)
    # print(out.shape)
    #
    # iam_module = IterativeAlignmentUnit(num_feat=64, kernel_size=(3, 3), deformable_groups=8).to(device)
    # nbr_fea = torch.randn(1, 64, 128, 128).to(device)
    # ref_fea = torch.randn(1, 64, 128, 128).to(device)
    # out1, out2 = iam_module(nbr_fea, ref_fea)
    # print(out1.shape)
    # print(out2.shape)

    corr = LocalCorr(nf=64, nbr_size=9).to(device)
    # nbr_list = [
    #     torch.randn(1, 64, 128, 128).to(device),
    #     torch.randn(1, 64, 128, 128).to(device)
    # ]
    nbr_fea = torch.randn(1, 64, 128, 128).to(device)
    ref_fea = torch.randn(1, 64, 128, 128).to(device)
    al_fea = corr(nbr_fea, ref_fea)
    print(al_fea.shape)
