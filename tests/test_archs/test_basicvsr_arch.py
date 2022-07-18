import torch

from basicsr.archs.arch_util import FirstOrderDeformableAlignment, SecondOrderDeformableAlignment
from basicsr.archs.basicvsr_arch import BasicVSR, ConvResidualBlocks, IconVSR
from basicsr.archs.basicvsrplusplus_arch import BasicVSRPlusPlus


def test_basicvsr():
    """Test arch: BasicVSR."""

    # model init and forward
    net = BasicVSR(num_feat=12, num_block=2, spynet_path=None).cuda()
    img = torch.rand((1, 2, 3, 64, 64), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 2, 3, 256, 256)


def test_convresidualblocks():
    """Test block: ConvResidualBlocks."""

    # model init and forward
    net = ConvResidualBlocks(num_in_ch=3, num_out_ch=8, num_block=2).cuda()
    img = torch.rand((1, 3, 16, 16), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 8, 16, 16)


def test_iconvsr():
    """Test arch: IconVSR."""

    # model init and forward
    net = IconVSR(
        num_feat=8, num_block=1, keyframe_stride=2, temporal_padding=2, spynet_path=None, edvr_path=None).cuda()
    img = torch.rand((1, 6, 3, 64, 64), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 6, 3, 256, 256)

    # --------------------------- temporal padding 3 ------------------------- #
    net = IconVSR(
        num_feat=8, num_block=1, keyframe_stride=2, temporal_padding=3, spynet_path=None, edvr_path=None).cuda()
    img = torch.rand((1, 8, 3, 64, 64), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 8, 3, 256, 256)


if __name__ == '__main__':

    device = torch.device('cuda')
    net = BasicVSRPlusPlus(num_feat=64,
                           num_blocks=7,
                           max_residue_magnitude=10,
                           is_low_res_input=True,
                           spynet_path=None,
                           cpu_cache_length=100).to(device)
    img = torch.randn(1, 10, 3, 64, 64).to(device)
    output = net(img)
    assert output.shape == (1, 10, 3, 256, 256)
