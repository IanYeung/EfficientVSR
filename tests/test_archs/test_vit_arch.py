import torch
from torch import nn as nn
from torch.nn import functional as F

from einops import rearrange

from basicsr.archs.restormer_arch import OverlapPatchEmbed, Attention, TransformerBlock
from basicsr.archs.restormer_arch import Restormer
from basicsr.ops.msda import MSDeformAttn


def main():
    device = torch.device('cuda')

    patch_embed = OverlapPatchEmbed(in_c=3, embed_dim=32, bias=False)
    attention = Attention(dim=32, num_heads=1, bias=True)
    trans_block = TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias')
    restormer = Restormer()

    inp = torch.randn(1, 3, 128, 128)

    out = patch_embed(inp)
    print(out.shape)

    out = attention(out)
    print(out.shape)

    out = trans_block(out)
    print(out.shape)


def accuracy_based_reweighting():

    device = torch.device('cuda')
    b, c, h, w = 1, 32, 64, 64

    x1 = torch.randn(b, c, h, w).to(device)
    x2 = torch.randn(b, c, h, w).to(device)

    y1 = F.unfold(x1, kernel_size=3, stride=1, padding=1).view(b, c, 3, 3, h, w)
    y2 = x2.unsqueeze(dim=2).unsqueeze(dim=3).expand(b, c, 3, 3, h, w)

    y1 = rearrange(y1, 'b c k1 k2 h w -> b h w c k1 k2')
    y2 = rearrange(y2, 'b c k1 k2 h w -> b h w c k1 k2')

    out = y1 @ y2.transpose(-2, -1)

    print(out.shape)


if __name__ == '__main__':

    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(spatial_shapes, valid_ratios, device):
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

    import torch
    from einops import rearrange

    device = torch.device('cuda')

    ms_deform_att = MSDeformAttn(d_model=32, n_levels=1, n_heads=8, n_points=4).to(device)
    nbr_fea = torch.randn(1, 32, 64, 64).to(device)
    ext_fea = torch.randn(1, 64, 64, 64).to(device)

    b, c, h, w = nbr_fea.shape

    #nbr_fea = rearrange(nbr_fea, 'b c h w -> b (h w) c')
    #ext_fea = rearrange(ext_fea, 'b c h w -> b (h w) c')
    mask = (torch.zeros(b, h, w) > 1).to(device)

    spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    valid_ratio = torch.unsqueeze(get_valid_ratio(mask), dim=1)
    ref_point = get_reference_points(spatial_shapes, valid_ratio, device=device)

    output = ms_deform_att(ext_fea, ref_point, nbr_fea, spatial_shapes, level_start_index,
                           input_padding_mask=mask.flatten(1))
    #output = rearrange(output, 'b (h w) c -> b c h w', h=h, w=w)
