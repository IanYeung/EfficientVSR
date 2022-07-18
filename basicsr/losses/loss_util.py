import functools
import torch
from torch.nn import functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper


def get_local_weights(batch_img, ksize):

    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    weighted_pix = torch.var(patches, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return weighted_pix


def cal_weight_map_with_ema_result(img_target, img_output, img_ema, ksize):

    diff_ema = torch.abs(img_target - img_ema)
    diff_SR = torch.abs(img_target - img_output)

    diff_SR = torch.sum(diff_SR, 1, keepdim=True)
    diff_ema = torch.sum(diff_ema, 1, keepdim=True)

    weight_global = torch.var(diff_SR.clone(), dim=(-1, -2, -3), keepdim=True) ** 0.2
    pixel_weight = get_local_weights(diff_SR.clone(), ksize)

    overall_weight = weight_global * pixel_weight
    overall_weight[diff_SR < diff_ema] = 0

    return overall_weight


def cal_weight_map_wout_ema_result(img_target, img_output, ksize):

    diff_SR = torch.abs(img_target - img_output)
    diff_SR = torch.sum(diff_SR, 1, keepdim=True)

    weight_global = torch.var(diff_SR.clone(), dim=(-1, -2, -3), keepdim=True) ** 0.2
    pixel_weight = get_local_weights(diff_SR.clone(), ksize)

    overall_weight = weight_global * pixel_weight

    return overall_weight
