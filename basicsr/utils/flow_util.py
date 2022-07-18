# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/video/optflow.py  # noqa: E501
import cv2
import numpy as np
import os


def flowread(flow_path, quantize=False, concat_axis=0, *args, **kwargs):
    """Read an optical flow map.

    Args:
        flow_path (ndarray or str): Flow path.
        quantize (bool): whether to read quantized pair, if set to True,
            remaining args will be passed to :func:`dequantize_flow`.
        concat_axis (int): The axis that dx and dy are concatenated,
            can be either 0 or 1. Ignored if quantize is False.

    Returns:
        ndarray: Optical flow represented as a (h, w, 2) numpy array
    """
    if quantize:
        assert concat_axis in [0, 1]
        cat_flow = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        if cat_flow.ndim != 2:
            raise IOError(f'{flow_path} is not a valid quantized flow file, its dimension is {cat_flow.ndim}.')
        assert cat_flow.shape[concat_axis] % 2 == 0
        dx, dy = np.split(cat_flow, 2, axis=concat_axis)
        flow = dequantize_flow(dx, dy, *args, **kwargs)
    else:
        with open(flow_path, 'rb') as f:
            try:
                header = f.read(4).decode('utf-8')
            except Exception:
                raise IOError(f'Invalid flow file: {flow_path}')
            else:
                if header != 'PIEH':
                    raise IOError(f'Invalid flow file: {flow_path}, ' 'header does not contain PIEH')

            w = np.fromfile(f, np.int32, 1).squeeze()
            h = np.fromfile(f, np.int32, 1).squeeze()
            flow = np.fromfile(f, np.float32, w * h * 2).reshape((h, w, 2))

    return flow.astype(np.float32)


def flowwrite(flow, filename, quantize=False, concat_axis=0, *args, **kwargs):
    """Write optical flow to file.

    If the flow is not quantized, it will be saved as a .flo file losslessly,
    otherwise a jpeg image which is lossy but of much smaller size. (dx and dy
    will be concatenated horizontally into a single image if quantize is True.)

    Args:
        flow (ndarray): (h, w, 2) array of optical flow.
        filename (str): Output filepath.
        quantize (bool): Whether to quantize the flow and save it to 2 jpeg
            images. If set to True, remaining args will be passed to
            :func:`quantize_flow`.
        concat_axis (int): The axis that dx and dy are concatenated,
            can be either 0 or 1. Ignored if quantize is False.
    """
    if not quantize:
        with open(filename, 'wb') as f:
            f.write('PIEH'.encode('utf-8'))
            np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
            flow = flow.astype(np.float32)
            flow.tofile(f)
            f.flush()
    else:
        assert concat_axis in [0, 1]
        dx, dy = quantize_flow(flow, *args, **kwargs)
        dxdy = np.concatenate((dx, dy), axis=concat_axis)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, dxdy)


def quantize_flow(flow, max_val=0.02, norm=True):
    """Quantize flow to [0, 255].

    After this step, the size of flow will be much smaller, and can be
    dumped as jpeg images.

    Args:
        flow (ndarray): (h, w, 2) array of optical flow.
        max_val (float): Maximum value of flow, values beyond
                        [-max_val, max_val] will be truncated.
        norm (bool): Whether to divide flow values by image width/height.

    Returns:
        tuple[ndarray]: Quantized dx and dy.
    """
    h, w, _ = flow.shape
    dx = flow[..., 0]
    dy = flow[..., 1]
    if norm:
        dx = dx / w  # avoid inplace operations
        dy = dy / h
    # use 255 levels instead of 256 to make sure 0 is 0 after dequantization.
    flow_comps = [quantize(d, -max_val, max_val, 255, np.uint8) for d in [dx, dy]]
    return tuple(flow_comps)


def dequantize_flow(dx, dy, max_val=0.02, denorm=True):
    """Recover from quantized flow.

    Args:
        dx (ndarray): Quantized dx.
        dy (ndarray): Quantized dy.
        max_val (float): Maximum value used when quantizing.
        denorm (bool): Whether to multiply flow values with width/height.

    Returns:
        ndarray: Dequantized flow.
    """
    assert dx.shape == dy.shape
    assert dx.ndim == 2 or (dx.ndim == 3 and dx.shape[-1] == 1)

    dx, dy = [dequantize(d, -max_val, max_val, 255) for d in [dx, dy]]

    if denorm:
        dx *= dx.shape[1]
        dy *= dx.shape[0]
    flow = np.dstack((dx, dy))
    return flow


def quantize(arr, min_val, max_val, levels, dtype=np.int64):
    """Quantize an array of (-inf, inf) to [0, levels-1].

    Args:
        arr (ndarray): Input array.
        min_val (scalar): Minimum value to be clipped.
        max_val (scalar): Maximum value to be clipped.
        levels (int): Quantization levels.
        dtype (np.type): The type of the quantized array.

    Returns:
        tuple: Quantized array.
    """
    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(f'levels must be a positive integer, but got {levels}')
    if min_val >= max_val:
        raise ValueError(f'min_val ({min_val}) must be smaller than max_val ({max_val})')

    arr = np.clip(arr, min_val, max_val) - min_val
    quantized_arr = np.minimum(np.floor(levels * arr / (max_val - min_val)).astype(dtype), levels - 1)

    return quantized_arr


def dequantize(arr, min_val, max_val, levels, dtype=np.float64):
    """Dequantize an array.

    Args:
        arr (ndarray): Input array.
        min_val (scalar): Minimum value to be clipped.
        max_val (scalar): Maximum value to be clipped.
        levels (int): Quantization levels.
        dtype (np.type): The type of the dequantized array.

    Returns:
        tuple: Dequantized array.
    """
    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(f'levels must be a positive integer, but got {levels}')
    if min_val >= max_val:
        raise ValueError(f'min_val ({min_val}) must be smaller than max_val ({max_val})')

    dequantized_arr = (arr + 0.5).astype(dtype) * (max_val - min_val) / levels + min_val

    return dequantized_arr


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)
