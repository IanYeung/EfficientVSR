import functools
from collections import OrderedDict
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from torch.nn import PixelShuffle, PixelUnshuffle
from basicsr.archs.arch_util import make_layer, default_init_weights, ResidualBlockNoBN
from basicsr.utils.registry import ARCH_REGISTRY


# FRVSR
def initialize_weights(net_l, init_type='kaiming', scale=1):
    """ Modify from BasicSR/MMSR
    """

    if not isinstance(net_l, list):
        net_l = [net_l]

    for net in net_l:
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                else:
                    raise NotImplementedError(init_type)

                m.weight.data *= scale  # to stabilize training

                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)


def space_to_depth(x, scale):
    """ Equivalent to tf.space_to_depth()
    """

    n, c, in_h, in_w = x.size()
    out_h, out_w = in_h // scale, in_w // scale

    x_reshaped = x.reshape(n, c, out_h, scale, out_w, scale)
    x_reshaped = x_reshaped.permute(0, 3, 5, 1, 2, 4)
    output = x_reshaped.reshape(n, scale * scale * c, out_h, out_w)

    return output


def backward_warp(x, flow, mode='bilinear', padding_mode='border'):
    """ Backward warp `x` according to `flow`

        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat([
        flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
        flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], dim=1)

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default for PyTorch version < 1.4.0
    if int(''.join(torch.__version__.split('.')[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output


def get_upsampling_func(scale=4, degradation='BI'):
    if degradation == 'BI':
        upsample_func = functools.partial(
            F.interpolate, scale_factor=scale, mode='bilinear',
            align_corners=False)

    elif degradation == 'BD':
        upsample_func = BicubicUpsampler(scale_factor=scale)

    else:
        raise ValueError(f'Unrecognized degradation type: {degradation}')

    return upsample_func


class BicubicUpsampler(nn.Module):
    """ Bicubic upsampling function with similar behavior to that in TecoGAN-Tensorflow

        Note:
            This function is different from torch.nn.functional.interpolate and matlab's imresize
            in terms of the bicubic kernel and the sampling strategy

        References:
            http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
            https://stackoverflow.com/questions/26823140/imresize-trying-to-understand-the-bicubic-interpolation
    """

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsampler, self).__init__()

        # calculate weights (according to Eq.(6) in the reference paper)
        cubic = torch.FloatTensor([
            [0, a, -2*a, a],
            [1, 0, -(a + 3), a + 2],
            [0, -a, (2*a + 3), -(a + 2)],
            [0, 0, a, -a]
        ])

        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s**2, s**3]))
            for s in [1.0*d/scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer('kernels', torch.stack(kernels))  # size: (f, 4)

    def forward(self, input):
        n, c, h, w = input.size()
        f = self.scale_factor

        # merge n&c
        input = input.reshape(n*c, 1, h, w)

        # pad input (left, right, top, bottom)
        input = F.pad(input, (1, 2, 1, 2), mode='replicate')

        # calculate output (vertical expansion)
        kernel_h = self.kernels.view(f, 1, 4, 1)
        output = F.conv2d(input, kernel_h, stride=1, padding=0)
        output = output.permute(0, 2, 1, 3).reshape(n*c, 1, f*h, w + 3)

        # calculate output (horizontal expansion)
        kernel_w = self.kernels.view(f, 1, 1, 4)
        output = F.conv2d(output, kernel_w, stride=1, padding=0)
        output = output.permute(0, 2, 3, 1).reshape(n*c, 1, f*h, f*w)

        # split n&c
        output = output.reshape(n, c, f*h, f*w)

        return output


class FNet(nn.Module):
    """ Optical flow estimation network
    """

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2*in_nc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True))

    def forward(self, x1, x2):
        """ Compute optical flow from x1 to x2
        """

        out = self.encoder1(torch.cat([x1, x2], dim=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = F.interpolate(
            self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity

        return out


class ResidualBlock(nn.Module):
    """ Residual block without batch normalization
    """

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

    def forward(self, x):
        out = self.conv(x) + x

        return out


class SRNet(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc, out_nc, nf, nb, upsample_func, scale):
        super(SRNet, self).__init__()

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling blocks
        conv_up = [
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True)]

        if scale == 4:
            conv_up += [
                nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
                nn.ReLU(inplace=True)]

        self.conv_up = nn.Sequential(*conv_up)

        # output conv.
        self.conv_out = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # upsampling function
        self.upsample_func = upsample_func

    def forward(self, lr_curr, hr_prev_tran):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(s*s*c)hw
        """

        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        out = self.resblocks(out)
        out = self.conv_up(out)
        out = self.conv_out(out)
        out += self.upsample_func(lr_curr)

        return out


# @ARCH_REGISTRY.register()
class FRVSR(nn.Module):
    """ Frame-recurrent network: https://arxiv.org/abs/1801.04590
    """

    def __init__(self, in_nc, out_nc, nf, nb, degradation, scale):
        super(FRVSR, self).__init__()

        self.scale = scale

        # get upsampling function according to degradation type
        self.upsample_func = get_upsampling_func(self.scale, degradation)

        # define fnet & srnet
        self.fnet = FNet(in_nc)
        self.srnet = SRNet(in_nc, out_nc, nf, nb, self.upsample_func, self.scale)

    def forward(self, lr_data):
        """
            Parameters:
                :param lr_data: lr data in shape ntchw
        """

        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

        # calculate optical flows
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_flow = self.fnet(lr_curr, lr_prev)  # n*(t-1),2,h,w

        # upsample lr flows
        hr_flow = self.scale * self.upsample_func(lr_flow)
        hr_flow = hr_flow.view(n, (t - 1), 2, hr_h, hr_w)

        # compute the first hr data
        hr_data = []
        hr_prev = self.srnet(
            lr_data[:, 0, ...],
            torch.zeros(n, (self.scale**2)*c, lr_h, lr_w, dtype=torch.float32, device=lr_data.device)
        )
        hr_data.append(hr_prev)

        # compute the remaining hr data
        for i in range(1, t):
            # warp hr_prev
            hr_prev_warp = backward_warp(hr_prev, hr_flow[:, i - 1, ...])

            # compute hr_curr
            hr_curr = self.srnet(
                lr_data[:, i, ...],
                space_to_depth(hr_prev_warp, self.scale))

            # save and update
            hr_data.append(hr_curr)
            hr_prev = hr_curr

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w

        if self.training:
            # construct output dict
            ret_dict = {
                'hr_data': hr_data,  # n,t,c,hr_h,hr_w
                'hr_flow': hr_flow,  # n,t,2,hr_h,hr_w
                'lr_prev': lr_prev,  # n(t-1),c,lr_h,lr_w
                'lr_curr': lr_curr,  # n(t-1),c,lr_h,lr_w
                'lr_flow': lr_flow,  # n(t-1),2,lr_h,lr_w
            }
            return ret_dict
        else:
            return hr_data

    def step(self, lr_curr, lr_prev, hr_prev):
        """
            Parameters:
                :param lr_curr: the current lr data in shape nchw
                :param lr_prev: the previous lr data in shape nchw
                :param hr_prev: the previous hr data in shape nc(sh)(sw)
        """

        # estimate lr flow (lr_curr -> lr_prev)
        lr_flow = self.fnet(lr_curr, lr_prev)

        # pad if size is not a multiple of 8
        pad_h = lr_curr.size(2) - lr_curr.size(2)//8*8
        pad_w = lr_curr.size(3) - lr_curr.size(3)//8*8
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), 'reflect')

        # upsample lr flow
        hr_flow = self.scale * self.upsample_func(lr_flow_pad)

        # warp hr_prev
        hr_prev_warp = backward_warp(hr_prev, hr_flow)

        # compute hr_curr
        hr_curr = self.srnet(lr_curr, space_to_depth(hr_prev_warp, self.scale))

        return hr_curr


# RLSP
def shuffle_down(x, factor):
    # format: (B, C, H, W)
    b, c, h, w = x.shape

    assert h % factor == 0 and w % factor == 0, "H and W must be a multiple of " + str(factor) + "!"

    n = x.reshape(b, c, int(h/factor), factor, int(w/factor), factor)
    n = n.permute(0, 3, 5, 1, 2, 4)
    n = n.reshape(b, c*factor**2, int(h/factor), int(w/factor))

    return n


def shuffle_up(x, factor):
    # format: (B, C, H, W)
    b, c, h, w = x.shape

    assert c % factor**2 == 0, "C must be a multiple of " + str(factor**2) + "!"

    n = x.reshape(b, factor, factor, int(c/(factor**2)), h, w)
    n = n.permute(0, 3, 4, 1, 5, 2)
    n = n.reshape(b, int(c/(factor**2)), factor*h, factor*w)

    return n


# @ARCH_REGISTRY.register()
class RLSP(nn.Module):

    def __init__(self, filters=128, state_dim=128, layers=7, kernel_size=3, factor=4):
        super(RLSP, self).__init__()

        self.factor = factor
        self.filters = filters
        self.kernel_size = kernel_size
        self.layers = layers
        self.state_dim = state_dim

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(3 * 3 + 3 * factor ** 2 + state_dim, filters, kernel_size, padding=int(kernel_size / 2))
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(filters, filters, kernel_size, padding=int(kernel_size / 2)) for _ in range(layers - 2)]
        )
        self.conv_out = nn.Conv2d(filters, 3 * factor ** 2 + state_dim, kernel_size, padding=int(kernel_size / 2))

    def cell(self, x, fb, state):

        factor = self.factor

        res = x[:, 1]  # keep x for residual connection

        input = torch.cat([x[:, 0], x[:, 1], x[:, 2], shuffle_down(fb, factor), state], -3)

        # first convolution
        x = self.act(self.conv1(input))

        # main convolution block
        for layer in self.conv_list:
            x = self.act(layer(x))

        x = self.conv_out(x)

        out = shuffle_up(x[..., :3 * factor ** 2, :, :] + res.repeat(1, factor ** 2, 1, 1), factor)
        state = self.act(x[..., 3 * factor ** 2:, :, :])

        return out, state

    def forward(self, x):

        factor = self.factor
        state_dim = self.state_dim

        seq = []
        for i in range(x.shape[1]):
            if i == 0:
                out = shuffle_up(torch.zeros_like(x[:, 0]).repeat(1, factor ** 2, 1, 1), factor)
                state = torch.zeros_like(x[:, 0, 0:1, ...]).repeat(1, state_dim, 1, 1)
                out, state = self.cell(torch.cat([x[:, i:i + 1], x[:, i:i + 2]], 1), out, state)
            elif i == x.shape[1] - 1:
                out, state = self.cell(torch.cat([x[:, i - 1:i + 1], x[:, i:i + 1]], 1), out, state)
            else:
                out, state = self.cell(x[:, i - 1:i + 2], out, state)
            seq.append(out)

        seq = torch.stack(seq, 1)

        return seq


# RRN
class Neuron(nn.Module):

    def __init__(self, n_c, n_b, scale):
        super(Neuron, self).__init__()
        pad = (1, 1)
        self.conv_1 = nn.Conv2d(scale ** 2 * 3 + n_c + 3 * 2, n_c, (3, 3), stride=(1, 1), padding=pad)
        basic_block = functools.partial(ResidualBlockNoBN, num_feat=n_c)
        self.recon_trunk = make_layer(basic_block, n_b)
        self.conv_h = nn.Conv2d(n_c, n_c, (3, 3), stride=(1, 1), padding=pad)
        self.conv_o = nn.Conv2d(n_c, scale ** 2 * 3, (3, 3), stride=(1, 1), padding=pad)
        default_init_weights([self.conv_1, self.conv_h, self.conv_o], 0.1)

    def forward(self, x, h, o):
        x = torch.cat((x, h, o), dim=1)
        x = F.relu(self.conv_1(x))
        x = self.recon_trunk(x)
        x_h = F.relu(self.conv_h(x))
        x_o = self.conv_o(x)
        return x_h, x_o


# @ARCH_REGISTRY.register()
class RRN(nn.Module):

    def __init__(self, n_c=128, n_b=10, scale=4):
        super(RRN, self).__init__()
        self.neuro = Neuron(n_c, n_b, scale)
        self.scale = scale
        self.down = PixelUnshuffle(scale)
        self.n_c = n_c

    def process(self, x, x_h, x_o, init):
        _, _, T, _, _ = x.shape
        f1 = x[:, 0, :, :, :]
        f2 = x[:, 1, :, :, :]
        x_input = torch.cat((f1, f2), dim=1)
        if init:
            x_h, x_o = self.neuro(x_input, x_h, x_o)
        else:
            x_o = self.down(x_o)
            x_h, x_o = self.neuro(x_input, x_h, x_o)
        x_o = F.pixel_shuffle(x_o, self.scale) + \
              F.interpolate(f2, scale_factor=self.scale, mode='bilinear', align_corners=False)

        return x_h, x_o

    def forward(self, x_input):
        B, T, C, H, W = x_input.shape
        scale = self.scale
        n_c = self.n_c
        out = []
        init = True

        for i in range(T):
            if init:
                init_i = x_input.new_zeros(B, C, H, W)
                init_o = x_input.new_zeros(B, scale * scale * 3, H, W)
                init_h = x_input.new_zeros(B, n_c, H, W)
                net_input = torch.stack([init_i, x_input[:, i, :, :, :]], dim=1)
                h, prediction = self.process(net_input, init_h, init_o, init)
                out.append(prediction)
                init = False
            else:
                net_input = x_input[:, i-1:i+1, :, :, :]
                h, prediction = self.process(net_input, h, prediction, init)
                out.append(prediction)
        out = torch.stack(out, dim=1)
        return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp = torch.rand(1, 5, 3, 128, 128).to(device)

    model = FRVSR(in_nc=3, out_nc=3, nf=32, nb=10, degradation='BI', scale=4).to(device)
    model.eval()
    with torch.no_grad():
        out = model(inp)
        print(out.shape)

    model = RLSP().to(device)
    model.eval()
    with torch.no_grad():
        out = model(inp)
        print(out.shape)

    model = RRN().to(device)
    model.eval()
    with torch.no_grad():
        out = model(inp)
        print(out.shape)
