import torch
from torch.nn import functional as F

from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class PCDBasedDCNVSRModel(VideoBaseModel):

    def setup_optimizers(self):
        train_opt = self.opt['train']
        dcn_lr_mul = train_opt.get('dcn_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for dcn with {dcn_lr_mul}.')
        if dcn_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate dcn params and normal params for different lr
            normal_params = []
            dcn_params = []
            for name, param in self.net_g.named_parameters():
                if 'dcn' in name:
                    dcn_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': dcn_params,
                    'lr': train_opt['optim_g']['lr'] * dcn_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)


@MODEL_REGISTRY.register()
class FlowGuidedDCNVSRModel(VideoBaseModel):

    def __init__(self, opt):
        super(FlowGuidedDCNVSRModel, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for dcn and flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate dcn params and normal params for different lr
            normal_params = []
            dcn_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:  # or 'dcn' in name or 'deform_att' in name:
                    dcn_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': dcn_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        super(FlowGuidedDCNVSRModel, self).optimize_parameters(current_iter)


@MODEL_REGISTRY.register()
class PCDBasedDCNSwinVSRModel(PCDBasedDCNVSRModel):

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        assert mod_pad_h % 2 == 0
        assert mod_pad_w % 2 == 0
        img = F.pad(self.lq, (mod_pad_w // 2, mod_pad_w // 2, mod_pad_h // 2, mod_pad_h // 2, 0, 0), 'replicate')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = self.output.size()

        self.output = self.output[:, :,
                      mod_pad_h // 2 * scale:h - mod_pad_h // 2 * scale,
                      mod_pad_w // 2 * scale:w - mod_pad_w // 2 * scale]

    # def test(self):
    #     # pad to multiplication of window_size
    #     window_size = self.opt['network_g']['window_size']
    #     scale = self.opt.get('scale', 1)
    #     mod_pad_h, mod_pad_w = 0, 0
    #     _, _, _, h, w = self.lq.size()
    #     if h % window_size != 0:
    #         mod_pad_h = window_size - h % window_size
    #     if w % window_size != 0:
    #         mod_pad_w = window_size - w % window_size
    #     img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h, 0, 0), 'replicate')
    #     if hasattr(self, 'net_g_ema'):
    #         self.net_g_ema.eval()
    #         with torch.no_grad():
    #             self.output = self.net_g_ema(img)
    #     else:
    #         self.net_g.eval()
    #         with torch.no_grad():
    #             self.output = self.net_g(img)
    #         self.net_g.train()
    #
    #     _, _, h, w = self.output.size()
    #     self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]


@MODEL_REGISTRY.register()
class FlowGuidedDCNSwinVSRModel(FlowGuidedDCNVSRModel):

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        assert mod_pad_h % 2 == 0
        assert mod_pad_w % 2 == 0
        img = F.pad(self.lq, (mod_pad_w // 2, mod_pad_w // 2, mod_pad_h // 2, mod_pad_h // 2, 0, 0), 'replicate')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = self.output.size()

        self.output = self.output[:, :,
                      mod_pad_h // 2 * scale:h - mod_pad_h // 2 * scale,
                      mod_pad_w // 2 * scale:w - mod_pad_w // 2 * scale]
