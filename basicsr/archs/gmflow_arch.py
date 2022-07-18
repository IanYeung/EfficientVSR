import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode, return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


def global_correlation_softmax(feature0, feature1, pred_bidir_flow=False):
    # global correlation
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, H*W]

    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)  # [B, H, W, H, W]

    # flow from softmax
    init_grid = coords_grid(b, h, w).to(correlation.device)  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]

    if pred_bidir_flow:
        correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)  # [2*B, H*W, H*W]
        init_grid = init_grid.repeat(2, 1, 1, 1)  # [2*B, 2, H, W]
        grid = grid.repeat(2, 1, 1)  # [2*B, H*W, 2]
        b = b * 2

    prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]

    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid

    return flow, prob


def local_correlation_softmax(feature0, feature1, local_radius, padding_mode='zeros'):

    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]

    valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode=padding_mode, align_corners=True
                                   ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

    # mask invalid locations
    corr[~valid] = -1e9

    prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)^2]

    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax)\
        .squeeze(-2).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    flow = correspondence - coords_init
    match_prob = prob

    return flow, match_prob


if __name__ == '__main__':
    from basicsr.archs.arch_util import FGAC

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # feature0 = torch.rand(1, 32, 128, 128).to(device)
    # feature1 = torch.rand(1, 32, 128, 128).to(device)
    # flow, _ = global_correlation_softmax(feature0, feature1)
    # print(flow.shape)

    # ref = torch.rand(1, 32, 128, 128).to(device)
    # sou = torch.rand(1, 32, 128, 128).to(device)
    # flow_s2r = torch.rand(1, 2, 128, 128).to(device)
    # fgac = FGAC(nf=32).to(device)
    # out, _, _ = fgac(ref, sou, flow_s2r)
    # print(out.shape)

    b, c, h, w = 1, 32, 2160 // 4 // 8, 3840 // 4 // 8

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros((repetitions, 1))

    # # GPU-WARM-UP
    # for _ in range(10):
    #     dummy_input1 = torch.randn(b, c, h, w).to(device)
    #     dummy_input2 = torch.randn(b, c, h, w).to(device)
    #     flow, _ = global_correlation_softmax(dummy_input1, dummy_input2)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            dummy_input1 = torch.randn(b, c, h, w).to(device)
            dummy_input2 = torch.randn(b, c, h, w).to(device)
            flow_s2r = torch.rand(b, 2, h, w).to(device)
            starter.record()
            # out, _, _ = fgac(dummy_input1, dummy_input2, flow_s2r)
            flow, _ = global_correlation_softmax(dummy_input1, dummy_input2, pred_bidir_flow=False)
            # flow, _ = local_correlation_softmax(dummy_input1, dummy_input2, local_radius=10)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)
