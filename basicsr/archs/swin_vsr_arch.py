import torch
from torch import nn as nn
from torch.nn import functional as F

from einops import rearrange

from basicsr.archs.arch_util import make_layer, flow_warp, trunc_normal_, ResidualBlockNoBN, \
    FirstOrderDeformableAlignment
from basicsr.archs.edvr_arch import PCDAlignment
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.swinir_arch import PatchEmbed, PatchUnEmbed, RSTB
from basicsr.archs.swinir_fast_arch import SwinBlock
from basicsr.archs.vswintransformer_arch import BasicLayer as SwinFusion
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class PCDAlignSwinVSRv1(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 center_frame_idx=None,
                 img_size=64,
                 patch_size=1,
                 depths=(2, 2, 2, 2, 2),
                 num_heads=(2, 2, 2, 2, 2),
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv'):
        super(PCDAlignSwinVSRv1, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)

        # fusion
        self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=nn.LayerNorm if patch_norm else None)

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=nn.LayerNorm if patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.num_layers = len(depths)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=num_feat,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(num_feat)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        b, t, c, h, w = x.size()
        assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(),
                feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.forward_features(feat)

        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        return out


@ARCH_REGISTRY.register()
class PCDAlignSwinVSRv2(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 center_frame_idx=None,
                 img_size=64,
                 patch_size=1,
                 depths=(2, 2, 2, 2, 2),
                 num_heads=(2, 2, 2, 2, 2),
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv',
                 fusion_depth=2,
                 fusion_num_heads=2):
        super(PCDAlignSwinVSRv2, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)

        # fusion
        self.spatial_temporal_att = SwinFusion(dim=num_feat,
                                               depth=fusion_depth,
                                               num_heads=fusion_num_heads,
                                               window_size=(num_frame, window_size, window_size))
        self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=nn.LayerNorm if patch_norm else None)

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=nn.LayerNorm if patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.num_layers = len(depths)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=num_feat,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(num_feat)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        b, t, c, h, w = x.size()
        assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(),
                feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        aligned_feat = rearrange(aligned_feat, 'b t c h w -> b c t h w')
        aligned_feat = self.spatial_temporal_att(aligned_feat)
        aligned_feat = rearrange(aligned_feat, 'b c t h w -> b t c h w')

        aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.forward_features(feat)

        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        return out


@ARCH_REGISTRY.register()
class FlowGuidedDCNSwinVSRv1(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 center_frame_idx=None,
                 spynet_path=None,
                 img_size=64,
                 patch_size=1,
                 depths=(2, 2, 2, 2, 2),
                 num_heads=(2, 2, 2, 2, 2),
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv'):
        super(FlowGuidedDCNSwinVSRv1, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        # flownet
        self.spynet = SpyNet(load_path=spynet_path)
        self.flow_guided_dcn = FirstOrderDeformableAlignment(num_feat,
                                                             num_feat,
                                                             3,
                                                             stride=1,
                                                             padding=1,
                                                             dilation=1,
                                                             groups=1,
                                                             deformable_groups=deformable_groups,
                                                             bias=True)

        # fusion
        self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=nn.LayerNorm if patch_norm else None)

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=nn.LayerNorm if patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.num_layers = len(depths)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=num_feat,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(num_feat)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

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
            cen = feat_l1[:, self.center_frame_idx, :, :, :].contiguous()
            nbr = feat_l1[:, i, :, :, :].contiguous()
            extra_feat = torch.cat([cen, flow_warp(nbr, flow.permute(0, 2, 3, 1))], dim=1)
            aligned_feat.append(self.flow_guided_dcn(nbr, extra_feat, flow))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.forward_features(feat)

        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        return out


@ARCH_REGISTRY.register()
class FlowGuidedDCNSwinVSRv2(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 center_frame_idx=None,
                 spynet_path=None,
                 img_size=64,
                 patch_size=1,
                 depths=(2, 2, 2, 2, 2),
                 num_heads=(2, 2, 2, 2, 2),
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv',
                 fusion_depth=2,
                 fusion_num_heads=2):
        super(FlowGuidedDCNSwinVSRv2, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        # flownet
        self.spynet = SpyNet(load_path=spynet_path)
        self.flow_guided_dcn = FirstOrderDeformableAlignment(num_feat,
                                                             num_feat,
                                                             3,
                                                             stride=1,
                                                             padding=1,
                                                             dilation=1,
                                                             groups=1,
                                                             deformable_groups=deformable_groups,
                                                             bias=True)

        # fusion
        self.spatial_temporal_att = SwinFusion(dim=num_feat,
                                               depth=fusion_depth,
                                               num_heads=fusion_num_heads,
                                               window_size=(num_frame, window_size, window_size))
        self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=nn.LayerNorm if patch_norm else None)

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=nn.LayerNorm if patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.num_layers = len(depths)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=num_feat,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(num_feat)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

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
            cen = feat_l1[:, self.center_frame_idx, :, :, :].contiguous()
            nbr = feat_l1[:, i, :, :, :].contiguous()
            extra_feat = torch.cat([cen, flow_warp(nbr, flow.permute(0, 2, 3, 1))], dim=1)
            aligned_feat.append(self.flow_guided_dcn(nbr, extra_feat, flow))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        aligned_feat = rearrange(aligned_feat, 'b t c h w -> b c t h w')
        aligned_feat = self.spatial_temporal_att(aligned_feat)
        aligned_feat = rearrange(aligned_feat, 'b c t h w -> b t c h w')

        aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.forward_features(feat)

        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        return out


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch)
        )

    def forward(self, fea):
        return self.main(fea)


class SwinBlocks(nn.Module):
    """Swin Blocks.

    Args:
        num_feat (int): Number of output channels. Default: 64.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_feat=64,
                 img_size=64,
                 patch_size=1,
                 depths=(2, 2, 2, 2, 2),
                 num_heads=(2, 2, 2, 2, 2),
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv'):
        super().__init__()

        self.conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=nn.LayerNorm if patch_norm else None)

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_feat,
            embed_dim=num_feat,
            norm_layer=nn.LayerNorm if patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.num_layers = len(depths)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=num_feat,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(num_feat)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        fea = self.conv(x)
        out = self.forward_features(fea)
        return out


class FastBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=64,
                 num_heads=4,
                 window_size=8,
                 mlp_ratio=2,
                 num_blocks=20,
                 repeats=4):
        super().__init__()
        assert num_blocks % repeats == 0
        self.conv = nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        body = []
        for i in range(num_blocks // repeats):
            body.append(
                SwinBlock(inp_channels=num_out_ch, out_channels=num_out_ch, repeats=repeats,
                          window_size=window_size, exp_ratio=mlp_ratio, heads=num_heads)
            )

        self.body = nn.Sequential(*body)

    def forward(self, fea):
        fea = self.relu(self.conv(fea))
        out = self.body(fea)
        return out


@ARCH_REGISTRY.register()
class BasicSwinVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_feat=64,
                 img_size=64,
                 window_size=8,
                 depths=(2, 2, 2, 2, 2),
                 num_heads=(2, 2, 2, 2, 2),
                 spynet_path=None,
                 return_flow=False):
        super().__init__()
        self.num_feat = num_feat
        self.return_flow = return_flow

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = SwinBlocks(num_in_ch=num_feat + 3, num_feat=num_feat, img_size=img_size,
                                         window_size=window_size, depths=depths, num_heads=num_heads)
        self.forward_trunk = SwinBlocks(num_in_ch=num_feat + 3, num_feat=num_feat, img_size=img_size,
                                        window_size=window_size, depths=depths, num_heads=num_heads)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

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
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        if self.return_flow:
            return torch.stack(out_l, dim=1), flows_forward, flows_backward
        else:
            return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class BasicSwinVSRCoupleProp(nn.Module):
    """BasicVSRCoupleProp.

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
                 img_size=64,
                 window_size=8,
                 depths=(2, 2, 2, 2, 2),
                 num_heads=(2, 2, 2, 2, 2),
                 spynet_path=None):
        super().__init__()

        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = SwinBlocks(num_in_ch=num_feat + 3, num_feat=num_feat, img_size=img_size,
                                         window_size=window_size, depths=depths, num_heads=num_heads)

        self.forward_trunk = SwinBlocks(num_in_ch=2 * num_feat + 3, num_feat=num_feat, img_size=img_size,
                                        window_size=window_size, depths=depths, num_heads=num_heads)
        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

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
        b, n, _, h, w = x.size()

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class FastPCDAlignSwinVSR(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruction_blocks=15,
                 num_heads=4,
                 window_size=8,
                 mlp_ratio=2,
                 norm=True,
                 center_frame_idx=None):
        super(FastPCDAlignSwinVSR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)

        # fusion
        self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = SwinBlock(repeats=num_reconstruction_blocks,
                                        inp_channels=num_feat,
                                        out_channels=num_feat,
                                        window_size=window_size,
                                        exp_ratio=mlp_ratio,
                                        heads=num_heads,
                                        norm=norm)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b, t, c, h, w = x.size()
        assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(),
                feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
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
class FastFlowGuidedDCNSwinVSR(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruction_blocks=15,
                 num_heads=4,
                 window_size=8,
                 mlp_ratio=2,
                 norm=True,
                 center_frame_idx=None,
                 spynet_path=None):
        super(FastFlowGuidedDCNSwinVSR, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        # flownet
        self.spynet = SpyNet(load_path=spynet_path)
        self.flow_guided_dcn = FirstOrderDeformableAlignment(num_feat,
                                                             num_feat,
                                                             3,
                                                             stride=1,
                                                             padding=1,
                                                             dilation=1,
                                                             groups=1,
                                                             deformable_groups=deformable_groups,
                                                             bias=True)

        # fusion
        self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = SwinBlock(repeats=num_reconstruction_blocks,
                                        inp_channels=num_feat,
                                        out_channels=num_feat,
                                        window_size=window_size,
                                        exp_ratio=mlp_ratio,
                                        heads=num_heads,
                                        norm=norm)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

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
            cen = feat_l1[:, self.center_frame_idx, :, :, :].contiguous()
            nbr = feat_l1[:, i, :, :, :].contiguous()
            extra_feat = torch.cat([cen, flow_warp(nbr, flow.permute(0, 2, 3, 1))], dim=1)
            aligned_feat.append(self.flow_guided_dcn(nbr, extra_feat, flow))
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
class FastBasicSwinVSRCoupleProp(nn.Module):
    """BasicVSRCoupleProp.

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
                 window_size=8,
                 num_heads=4,
                 mlp_ratio=2,
                 num_blocks=16,
                 repeats=4,
                 spynet_path=None):
        super().__init__()

        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = FastBlocks(num_in_ch=num_feat + 3, num_out_ch=num_feat,
                                         num_blocks=num_blocks, repeats=repeats,
                                         window_size=window_size, mlp_ratio=mlp_ratio, num_heads=num_heads)

        self.forward_trunk = FastBlocks(num_in_ch=2 * num_feat + 3, num_out_ch=num_feat,
                                        num_blocks=num_blocks, repeats=repeats,
                                        window_size=window_size, mlp_ratio=mlp_ratio, num_heads=num_heads)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

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
        b, n, _, h, w = x.size()

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)


@ARCH_REGISTRY.register()
class FastBasicSwinVSRCouplePropV3(nn.Module):
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
                 window_size=8,
                 num_heads=4,
                 mlp_ratio=2,
                 num_extract_block=0,
                 num_blocks=16,
                 repeats=4,
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
        self.backward_trunk = FastBlocks(num_in_ch=2 * num_feat, num_out_ch=num_feat,
                                         num_blocks=num_blocks, repeats=repeats,
                                         window_size=window_size, mlp_ratio=mlp_ratio, num_heads=num_heads)

        self.forward_trunk = FastBlocks(num_in_ch=3 * num_feat, num_out_ch=num_feat,
                                        num_blocks=num_blocks, repeats=repeats,
                                        window_size=window_size, mlp_ratio=mlp_ratio, num_heads=num_heads)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

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
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)