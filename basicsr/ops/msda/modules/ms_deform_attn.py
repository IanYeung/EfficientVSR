# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_

from einops import rearrange

from ..functions import MSDeformAttnFunction


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        constant_(module.bias, bias)


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


def single_scale_deformable_attn_v1(value, value_spatial_shapes, sampling_locations):
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

    scale = embed_dims ** -0.5

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
    attention_weights = torch.softmax(attention_weights, dim=-1) * scale

    # (bs, num_queries, num_heads, num_points) ->
    # (bs, num_heads, num_queries, num_points) ->
    # (bs*num_heads, 1, num_queries, num_points)
    # attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries, num_points)

    # (bs*num_heads, embed_dims, num_queries, num_points) *
    # (bs*num_heads,          1, num_queries, num_points)
    output = (sampling_values * attention_weights).sum(-1)
    output = output.view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


def single_scale_deformable_attn_v2(value, sampling_locations):
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


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
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
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
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

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(2 * d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(2 * d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1,
                                                                                                              self.n_levels,
                                                                                                              self.n_points,
                                                                                                              1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1])
            )
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
            self.im2col_step
        )
        output = self.output_proj(output)
        return output


# class MSDeformAttnMyVersion(nn.Module):
#
#     def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4, max_residue_magnitude=10):
#         """
#         Multi-Scale Deformable Attention Module
#         :param d_model      hidden dimension
#         :param n_levels     number of feature levels
#         :param n_heads      number of attention heads
#         :param n_points     number of sampling points per attention head per feature level
#         """
#         super().__init__()
#         if d_model % n_heads != 0:
#             raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
#         _d_per_head = d_model // n_heads
#         # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
#         if not _is_power_of_2(_d_per_head):
#             warnings.warn(
#                 "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2,"
#                 "which is more efficient in our CUDA implementation."
#             )
#
#         self.im2col_step = 64
#
#         self.max_residue_magnitude = max_residue_magnitude
#
#         self.d_model = d_model
#         self.n_levels = n_levels
#         self.n_heads = n_heads
#         self.n_points = n_points
#
#         self.conv_offset = nn.Sequential(
#             nn.Conv2d(2 * d_model, 1 * d_model, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(1 * d_model, 1 * d_model, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(1 * d_model, 1 * d_model, 3, 1, 1),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(1 * d_model, n_heads * n_levels * n_points * 3, 3, 1, 1),
#         )
#         self.val_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)
#         self.out_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)
#
#         self.init_offset()
#
#     def init_offset(self):
#         constant_init(self.conv_offset[-1], val=0, bias=0)
#
#     def forward(self, query, reference_points, value, input_spatial_shapes, input_level_start_index,
#                 input_padding_mask=None, flow=None):
#
#         N, C, H, W = value.shape
#         assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == H * W
#
#         value = self.val_proj(value)
#         value = value.flatten(2).transpose(1, 2)
#         if input_padding_mask is not None:
#             value = value.masked_fill(input_padding_mask[..., None], float(0))
#         value = value.view(N, H * W, self.n_heads, self.d_model // self.n_heads)
#
#         out = self.conv_offset(query)
#         offset1, offset2, weights = torch.chunk(out, 3, dim=1)
#         offsets = torch.cat((offset1, offset2), dim=1)  # (B, Nh*Nl*Np*2, H, W)
#         weights = weights.view(N, self.n_heads, self.n_levels * self.n_points, H, W)
#         weights = torch.softmax(weights, dim=2)
#         weights = weights.view(N, self.n_heads * self.n_levels * self.n_points, H, W)  # (B, Nh*Nl*Np*1, H, W)
#
#         # clamp offsets
#         offsets = self.max_residue_magnitude * torch.tanh(offsets)
#         # flow guided
#         if flow is not None:
#             offsets = offsets + flow.flip(1).repeat(1, offsets.size(1) // 2, 1, 1)
#
#         offsets = offsets.flatten(2).transpose(1, 2)    # (B, H*W, Nh*Nl*Np*2)
#         weights = weights.flatten(2).transpose(1, 2)    # (B, H*W, Nh*Nl*Np*1)
#         offsets = offsets.view(N, H * W, self.n_heads, self.n_levels, self.n_points, 2).contiguous()  # (B, H*W, Nh, Nl, Np, 2)
#         weights = weights.view(N, H * W, self.n_heads, self.n_levels, self.n_points, 1).contiguous()  # (B, H*W, Nh, Nl, Np, 1)
#
#         # N, Len_q, n_heads, n_levels, n_points, 2
#         if reference_points.shape[-1] == 2:
#             offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
#             sampling_locations = reference_points[:, :, None, :, None, :] + \
#                                  offsets / offset_normalizer[None, None, None, :, None, :]
#         elif reference_points.shape[-1] == 4:
#             sampling_locations = reference_points[:, :, None, :, None, :2] + \
#                                  offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
#         else:
#             raise ValueError(
#                 'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1])
#             )
#
#         output = MSDeformAttnFunction.apply(
#             value, input_spatial_shapes, input_level_start_index, sampling_locations, weights, self.im2col_step
#         )
#
#         output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
#         output = self.out_proj(output)
#
#         return output


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
        self.val_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)
        self.out_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)

        self.init_offset()

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
            offsets = offsets + flow.repeat(1, offsets.size(1) // 2, 1, 1)

        offsets = offsets.flatten(2).transpose(1, 2)    # (B, H*W, Nh*Nl*Np*2)
        weights = weights.flatten(2).transpose(1, 2)    # (B, H*W, Nh*Nl*Np*1)
        offsets = offsets.view(N, H * W, self.n_heads, self.n_levels, self.n_points, 2).contiguous()  # (B, H*W, Nh, Nl, Np, 2)
        weights = weights.view(N, H * W, self.n_heads, self.n_levels, self.n_points, 1).contiguous()  # (B, H*W, Nh, Nl, Np, 1)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                                 offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(
                'Last dim of reference_points must be 2, but get {} instead.'.format(reference_points.shape[-1])
            )

        output = multi_scale_deformable_attn_pytorch(
            value, input_spatial_shapes, sampling_locations, weights
        )
        # output = torch.randn((N, H*W, C), device=value.device)

        output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
        output = self.out_proj(output)

        return output


class SingleScaleDeformAttnV1(nn.Module):

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
        # self.val_proj = nn.Conv2d(d_model, d_model, 3, 1, 1)
        self.val_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)
        self.out_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)

        self.init_offset()

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
        if self.max_residue_magnitude:
            offsets = self.max_residue_magnitude * torch.tanh(offsets)
        # flow guided
        if flow is not None:
            offsets = offsets + flow.repeat(1, offsets.size(1) // 2, 1, 1)

        offsets = offsets.flatten(2).transpose(1, 2)    # (B, H*W, Nh*Np*2)
        offsets = offsets.view(N, H * W, self.n_heads, self.n_points, 2).contiguous()  # (B, H*W, Nh, Np, 2)

        # N, Len_q, n_heads, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, :, None, :] + \
                                 offsets / offset_normalizer[None, None, :, None, :]
        else:
            raise ValueError(
                'Last dim of reference_points must be 2, but get {} instead.'.format(reference_points.shape[-1])
            )

        output = single_scale_deformable_attn_v1(
            value, input_spatial_shapes, sampling_locations
        )
        # output = torch.randn((N, H*W, C), device=value.device)

        output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
        output = self.out_proj(output)

        return output


class SingleScaleDeformAttnV2(nn.Module):

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
            nn.Conv2d(2 * d_model + 2, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, 1 * d_model, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(1 * d_model, n_heads * n_points * 2, 3, 1, 1),
        )
        # self.val_proj = nn.Conv2d(d_model, d_model, 3, 1, 1)
        self.val_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)
        self.out_proj = nn.Conv2d(d_model, d_model, 1, 1, 0)

        self.init_offset()

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

        offsets = self.conv_offset(torch.cat([query, flow], dim=1))

        # clamp offsets
        if self.max_residue_magnitude:
            offsets = self.max_residue_magnitude * torch.tanh(offsets)
        # flow guided
        if flow is not None:
            offsets = offsets + flow.repeat(1, offsets.size(1) // 2, 1, 1)

        offsets = offsets.flatten(2).transpose(1, 2)    # (B, H*W, Nh*Np*2)
        offsets = offsets.view(N, H * W, self.n_heads, self.n_points, 2).contiguous()  # (B, H*W, Nh, Np, 2)

        # N, Len_q, n_heads, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, :, None, :] + \
                                 offsets / offset_normalizer[None, None, :, None, :]
        else:
            raise ValueError(
                'Last dim of reference_points must be 2, but get {} instead.'.format(reference_points.shape[-1])
            )

        output = single_scale_deformable_attn_v1(
            value, input_spatial_shapes, sampling_locations
        )

        output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
        output = self.out_proj(output)

        return output


class SingleScaleDeformAttnV3(nn.Module):

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

        output = single_scale_deformable_attn_v2(
            value, sampling_locations
        )
        output = self.out_proj(output)

        return output