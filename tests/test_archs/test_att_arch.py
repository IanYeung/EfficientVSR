import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_

from einops import rearrange


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        constant_(module.bias, bias)


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


def single_scale_deformable_attn(value, sampling_locations):
    """CPU version of multi-scale deformable attention.
    Args:
        value (torch.Tensor): The value has shape
            (bs, num_heads, embed_dims//num_heads, h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs, num_heads, num_points, h, w, 2),
            the last dimension 2 represent (x, y).
    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, num_heads, embed_dims, h, w = value.shape
    _, _, num_points, _, _, _ = sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1

    # bs*num_heads, embed_dims, h, w
    value = value.view(bs * num_heads, embed_dims, h, w)

    # bs*num_heads, num_points, h, w, 2
    sampling_grids = sampling_grids.view(bs * num_heads, num_points, h, w, 2)

    sampling_value_list = []
    for i in range(num_points):
        sampling_value = F.grid_sample(
            value,
            sampling_grids[:, i, :, :, :],
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        sampling_value_list.append(sampling_value)
    # bs*num_heads, embed_dims, h, w, num_points
    sampling_values = torch.stack(sampling_value_list, dim=-1)

    # A: bs*num_heads, embed_dims, h, w, 1
    # B: bs*num_heads, embed_dims, h, w, num_points

    # (bs*num_heads, h, w, num_points, embed_dims) @ (bs*num_heads, h, w, embed_dims, 1) ->
    # (bs*num_heads, h, w, num_points, 1) ->
    # (bs*num_heads, 1, h, w, num_points)
    A = sampling_values.permute(0, 2, 3, 4, 1)
    B = value.unsqueeze(-1).permute(0, 2, 3, 1, 4)

    attention_weights = (A @ B).permute(0, 4, 1, 2, 3)
    attention_weights = torch.softmax(attention_weights, dim=-1)

    # (bs*num_heads, embed_dims, h, w, num_points) *
    # (bs*num_heads,          1, h, w, num_points)
    # ->
    # (bs*num_heads, embed_dims, h, w)
    output = (sampling_values * attention_weights).sum(-1)
    output = output.view(bs, num_heads * embed_dims, h, w)
    return output.contiguous()


def single_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations):
    """CPU version of multi-scale deformable attention.
    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs, num_queries, num_heads, num_points, 2),
            the last dimension 2 represent (x, y).
    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, num_keys, num_heads, embed_dims = value.shape
    H_, W_ = value_spatial_shapes[0, 0], value_spatial_shapes[0, 1]
    _, num_queries, num_heads, num_points, _ = sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1

    # bs, H_*W_, num_heads, embed_dims ->
    # bs, H_*W_, num_heads*embed_dims ->
    # bs, num_heads*embed_dims, H_*W_ ->
    # bs*num_heads, embed_dims, H_, W_
    value_l_ = value.flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
    # bs, num_queries, num_heads, num_points, 2 ->
    # bs, num_heads, num_queries, num_points, 2 ->
    # bs*num_heads, num_queries, num_points, 2
    sampling_grid_l_ = sampling_grids[:, :, :].transpose(1, 2).flatten(0, 1)
    # bs*num_heads, embed_dims, num_queries, num_points
    sampling_values = F.grid_sample(
        value_l_,
        sampling_grid_l_,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    # (bs, num_keys, num_heads, embed_dims//num_heads)
    # (bs*num_heads, num_keys, embed_dims) ->
    # (bs*num_heads, num_keys, embed_dims, 1)
    # (bs*num_heads, num_queries, num_points, embed_dims) @ (bs*num_heads, num_keys, embed_dims, 1) ->
    # (bs*num_heads, num_queries, num_points, 1) ->
    # (bs*num_heads, 1, num_queries, num_points)
    Q = value.permute(0, 2, 1, 3)
    Q = Q.reshape(bs * num_heads, num_keys, embed_dims, 1)
    attention_weights = (sampling_values.permute(0, 2, 3, 1) @ Q).permute(0, 3, 1, 2)
    attention_weights = torch.softmax(attention_weights, dim=-1)

    # (bs, num_queries, num_heads, num_points) ->
    # (bs, num_heads, num_queries, num_points) ->
    # (bs*num_heads, 1, num_queries, num_points)
    # attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries, num_points)

    # (bs*num_heads, embed_dims, num_queries, num_levels*num_points) *
    # (bs*num_heads,          1, num_queries, num_levels*num_points)
    output = (sampling_values * attention_weights).sum(-1)
    output = output.view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.
    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs, num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs, num_queries, num_heads, num_levels, num_points),
    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2)\
            .reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level]\
            .transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    # (bs*num_heads, num_keys, embed_dims) ->
    # (bs*num_heads, num_keys, embed_dims, 1)
    # (bs*num_heads, num_queries, num_levels*num_points, embed_dims) @ (bs*num_heads, num_keys, embed_dims, 1) ->
    # (bs*num_heads, num_queries, num_levels*num_points, 1) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)

    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)

    # (bs*num_heads, embed_dims, num_queries, num_levels*num_points) *
    # (bs*num_heads,          1, num_queries, num_levels*num_points)
    sampling_values = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_values * attention_weights).sum(-1)
    output = output.view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class MSDeformAttnMyVersion(nn.Module):

    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4, max_residue_magnitude=10):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2,"
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64

        self.max_residue_magnitude = max_residue_magnitude

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * d_model, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, n_heads * n_levels * n_points * 3, 3, 1, 1),
        )
        # self.val_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)
        self.out_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)

        # self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, query, reference_points, value, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None, flow=None):

        N, C, H, W = value.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == H * W

        # value = self.val_proj(value)
        value = value.flatten(2).transpose(1, 2)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, H * W, self.n_heads, self.d_model // self.n_heads)

        out = self.conv_offset(query)
        offset1, offset2, weights = torch.chunk(out, 3, dim=1)
        offsets = torch.cat((offset1, offset2), dim=1)  # (B, Nh*Nl*Np*2, H, W)
        weights = weights.view(N, self.n_heads, self.n_levels * self.n_points, H, W)
        weights = torch.softmax(weights, dim=2)
        weights = weights.view(N, self.n_heads * self.n_levels * self.n_points, H, W)  # (B, Nh*Nl*Np*1, H, W)

        # clamp offsets
        offsets = self.max_residue_magnitude * torch.tanh(offsets)
        # flow guided
        if flow is not None:
            offsets = offsets + flow.flip(1).repeat(1, offsets.size(1) // 2, 1, 1)

        offsets = offsets.flatten(2).transpose(1, 2)    # (B, H*W, Nh*Nl*Np*2)
        weights = weights.flatten(2).transpose(1, 2)    # (B, H*W, Nh*Nl*Np*1)
        offsets = offsets.view(N, H * W, self.n_heads, self.n_levels, self.n_points, 2).contiguous()  # (B, H*W, Nh, Nl, Np, 2)
        weights = weights.view(N, H * W, self.n_heads, self.n_levels, self.n_points, 1).contiguous()  # (B, H*W, Nh, Nl, Np, 1)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                                 offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + \
                                 offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1])
            )

        output = multi_scale_deformable_attn_pytorch(
            value, input_spatial_shapes, sampling_locations, weights
        )

        output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
        output = self.out_proj(output)

        return output


class MultiscaleDeformAttnAlign(nn.Module):

    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ms_deform_att = MSDeformAttnMyVersion(d_model=d_model,
                                                   n_levels=n_levels,
                                                   n_heads=n_heads,
                                                   n_points=n_points,
                                                   max_residue_magnitude=max_residue_magnitude)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
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

    def forward(self, nbr_fea, ext_fea):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        mask = (torch.zeros(b, h, w) > 1).to(device)

        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratio = torch.unsqueeze(self.get_valid_ratio(mask), dim=1)
        ref_point = self.get_reference_points(spatial_shapes, valid_ratio, device=device)

        output = self.ms_deform_att(ext_fea, ref_point, nbr_fea, spatial_shapes, level_start_index,
                                    input_padding_mask=mask.flatten(1), flow=None)

        return output


class FlowGuidedDeformAttnAlign(nn.Module):

    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ms_deform_att = MSDeformAttnMyVersion(d_model=d_model,
                                                   n_levels=n_levels,
                                                   n_heads=n_heads,
                                                   n_points=n_points,
                                                   max_residue_magnitude=max_residue_magnitude)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
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

    def forward(self, nbr_fea, ext_fea, flow):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        mask = (torch.zeros(b, h, w) > 1).to(device)

        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratio = torch.unsqueeze(self.get_valid_ratio(mask), dim=1)
        ref_point = self.get_reference_points(spatial_shapes, valid_ratio, device=device)

        output = self.ms_deform_att(ext_fea, ref_point, nbr_fea, spatial_shapes, level_start_index,
                                    input_padding_mask=mask.flatten(1), flow=flow)

        return output


class SSDeformAttnMyVersionV1(nn.Module):

    def __init__(self, d_model=256, n_heads=8, n_points=4, max_residue_magnitude=10):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2,"
                "which is more efficient in our CUDA implementation."
            )

        self.max_residue_magnitude = max_residue_magnitude

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * d_model, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, n_heads * n_points * 2, 3, 1, 1),
        )
        self.val_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)
        self.out_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)

        # self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, query, reference_points, value, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None, flow=None):

        N, C, H, W = value.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == H * W

        value = self.val_proj(value)
        value = value.flatten(2).transpose(1, 2)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, H * W, self.n_heads, self.d_model // self.n_heads)

        offsets = self.conv_offset(query)

        # clamp offsets
        offsets = self.max_residue_magnitude * torch.tanh(offsets)
        # flow guided
        if flow is not None:
            offsets = offsets + flow.flip(1).repeat(1, offsets.size(1) // 2, 1, 1)

        offsets = offsets.flatten(2).transpose(1, 2)    # (B, H*W, Nh*Np*2)
        offsets = offsets.view(N, H * W, self.n_heads, self.n_points, 2).contiguous()  # (B, H*W, Nh, Np, 2)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, :, None, :] + \
                                 offsets / offset_normalizer[None, None, :, None, :]
        else:
            raise ValueError(
                'Last dim of reference_points must be 2, but get {} instead.'.format(reference_points.shape[-1])
            )

        output = single_scale_deformable_attn_pytorch(
            value, input_spatial_shapes, sampling_locations
        )

        output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
        output = self.out_proj(output)

        return output


class FlowGuidedDeformAttnAlignV2(nn.Module):

    def __init__(self, d_model=256, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ms_deform_att = SSDeformAttnMyVersionV1(d_model=d_model,
                                                     n_heads=n_heads,
                                                     n_points=n_points,
                                                     max_residue_magnitude=max_residue_magnitude)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
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

    def forward(self, nbr_fea, ext_fea, flow):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        mask = (torch.zeros(b, h, w) > 1).to(device)

        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratio = torch.unsqueeze(self.get_valid_ratio(mask), dim=1)
        ref_point = self.get_reference_points(spatial_shapes, valid_ratio, device=device)

        output = self.ms_deform_att(ext_fea, ref_point, nbr_fea, spatial_shapes, level_start_index,
                                    input_padding_mask=mask.flatten(1), flow=flow)

        return output


class SSDeformAttnMyVersionV2(nn.Module):

    def __init__(self, d_model=256, n_heads=8, n_points=4, max_residue_magnitude=10):
        """
        Single-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2,"
                "which is more efficient in our CUDA implementation."
            )

        self.max_residue_magnitude = max_residue_magnitude

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * d_model, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, n_heads * n_points * 2, 3, 1, 1),
        )
        self.val_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)
        self.out_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, query, value, reference_points, flow=None):

        N, C, H, W = value.shape
        device = value.device

        value = self.val_proj(value)
        value = value.view(N, self.n_heads, self.d_model // self.n_heads, H, W)

        offsets = self.conv_offset(query)
        # clamp offsets
        if self.max_residue_magnitude:
            offsets = self.max_residue_magnitude * torch.tanh(offsets)
        # flow guided
        if flow is not None:
            offsets = offsets + flow.repeat(1, offsets.size(1) // 2, 1, 1)

        offsets = offsets.view(N, self.n_heads, self.n_points, H, W, 2).contiguous()  # (B, Nh, Np, H, W, 2)

        # (B, Nh, Np, h, w, 2)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([torch.tensor(W, dtype=torch.long, device=device),
                                             torch.tensor(H, dtype=torch.long, device=device)], -1)
            offset_normalizer = offset_normalizer[None, None, None, None, None, :]
            sampling_locations = reference_points[:, None, None, :, :, :] + offsets / offset_normalizer
        else:
            raise ValueError(
                'Last dim of reference_points must be 2, but get {} instead.'.format(reference_points.shape[-1])
            )

        output = single_scale_deformable_attn(
            value, sampling_locations
        )

        output = self.out_proj(output)

        return output


class FlowGuidedDeformAttnAlignV3(nn.Module):

    def __init__(self, d_model=256, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ss_deform_att = SSDeformAttnMyVersionV2(d_model=d_model,
                                                     n_heads=n_heads,
                                                     n_points=n_points,
                                                     max_residue_magnitude=max_residue_magnitude)

    def get_ref_points(self, H_, W_, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
        )
        ref = torch.stack((ref_x, ref_y), -1)
        ref[..., 0].div_(W_)  #.mul_(2).sub_(1)
        ref[..., 1].div_(H_)  #.mul_(2).sub_(1)
        ref = ref[None, ...]

        return ref

    def forward(self, nbr_fea, ext_fea, flow):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        ref_point = self.get_ref_points(h, w, device=device)
        output = self.ss_deform_att(ext_fea, nbr_fea, ref_point, flow=flow)

        return output


if __name__ == '__main__':

    # device = torch.device('cuda')
    # h, w = 128, 256
    # num_keys = h*w
    # bs, num_queries, num_heads, num_levels, num_points = 1, h*w, 4, 1, 9
    # embed_dims = 64
    #
    # value = torch.randn(bs, num_keys, embed_dims).view(bs, num_keys, num_heads, embed_dims // num_heads).to(device)
    # shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
    #
    # offset = torch.randn(bs, num_queries, num_heads, num_points, 2)
    # offset = torch.tanh(offset).to(device)
    # weight = torch.randn(bs, num_queries, num_heads, num_points).to(device)
    # out = single_scale_deformable_attn_pytorch(value, shapes, offset)
    #
    # # offset = torch.tanh(torch.randn(bs, num_queries, num_heads, num_levels, num_points, 2)).to(device)
    # # weight = torch.randn(bs, num_queries, num_heads, num_levels, num_points).to(device)
    # # out = multi_scale_deformable_attn_pytorch(value, shapes, offset, weight)
    #
    # print(out.shape)

    # device = torch.device('cuda')
    # inp1 = torch.randn(1, 32, 64, 64).to(device)
    # inp2 = torch.randn(1, 64, 64, 64).to(device)
    # flow = torch.randn(1, 2, 64, 64).to(device)
    # # att = MultiscaleDeformAttnAlign(d_model=32).to(device)
    # # out = att(inp1, inp2)
    # # print(out.shape)
    # att = FlowGuidedDeformAttnAlignV3(d_model=32).to(device)
    # out = att(inp1, inp2, flow)
    # print(out.shape)

    # device = torch.device('cuda')
    # bs, num_heads, embed_dims, h, w = 1, 4, 64, 128, 128
    # num_points = 9
    # inp = torch.randn(bs, num_heads, embed_dims // num_heads, h, w)
    # off = torch.tanh(torch.randn(bs, num_heads, num_points, h, w, 2))
    # out = single_scale_deformable_attn(inp, off)
    # print(out.shape)

    # def get_valid_ratio(mask):
    #     _, H, W = mask.shape
    #     valid_H = torch.sum(~mask[:, :, 0], 1)
    #     valid_W = torch.sum(~mask[:, 0, :], 1)
    #     valid_ratio_h = valid_H.float() / H
    #     valid_ratio_w = valid_W.float() / W
    #     valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    #     return valid_ratio
    #
    # def get_reference_points(spatial_shapes, valid_ratios, device):
    #     reference_points_list = []
    #     for lvl, (H_, W_) in enumerate(spatial_shapes):
    #         ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
    #                                       torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
    #         ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
    #         ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
    #         ref = torch.stack((ref_x, ref_y), -1)
    #         reference_points_list.append(ref)
    #     reference_points = torch.cat(reference_points_list, 1)
    #     reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    #     return reference_points
    #
    # def get_ref_points(H_, W_, device):
    #
    #     ref_y, ref_x = torch.meshgrid(
    #         torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
    #         torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
    #     )
    #     ref = torch.stack((ref_x, ref_y), -1)
    #     ref[..., 0].div_(W_)  #.mul_(2).sub_(1)
    #     ref[..., 1].div_(H_)  #.mul_(2).sub_(1)
    #     ref = ref[None, ...]
    #
    #     return ref
    #
    #
    # b, c, h, w = 1, 32, 64, 128
    # device = torch.device('cuda')
    #
    # mask = (torch.zeros(b, h, w) > 1).to(device)
    #
    # spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
    # level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    # valid_ratio = torch.unsqueeze(get_valid_ratio(mask), dim=1)
    # ref_point_1 = get_reference_points(spatial_shapes, valid_ratio, device=device)
    #
    # ref_point_2 = get_ref_points(h, w, device=device)
    #
    # out = torch.sum(ref_point_1 - ref_point_2.flatten(1, 2))


    class PyramidAttentionAlign(nn.Module):
        """Pyramid Attention Alignment.

        Args:
            num_feat (int): Channel number of intermediate features. Default: 64.
        """

        def __init__(self, num_feat=64):
            super(PyramidAttentionAlign, self).__init__()

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

    device = torch.device('cuda')
    nbr_fea = torch.randn(1, 64, 128, 128)
    ref_fea = torch.randn(1, 64, 128, 128)
    pyr_att = PyramidAttentionAlign(num_feat=64)
    out_fea = pyr_att(nbr_fea, ref_fea)
    print(out_fea.shape)
