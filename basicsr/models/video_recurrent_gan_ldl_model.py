import torch
from collections import OrderedDict

from basicsr.archs import build_network, arch_util
from basicsr.losses import build_loss, loss_util
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .video_recurrent_model import VideoRecurrentModel


@MODEL_REGISTRY.register()
class VideoRecurrentGANLDLModel(VideoRecurrentModel):

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # build network net_g with Exponential Moving Average (EMA)
            # net_g_ema only used for testing on one GPU and saving.
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        if train_opt['fix_flow']:
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:  # The fix_flow now only works for spynet.
                    flow_params.append(param)
                else:
                    normal_params.append(param)

            optim_params = [
                {  # add flow params first
                    'params': flow_params,
                    'lr': train_opt['lr_flow']
                },
                {
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
            ]
        else:
            optim_params = self.net_g.parameters()

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        if self.fix_flow_iter:
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        if self.ema_decay > 0:
            self.output_ema = self.net_g_ema(self.lq)

        b, n, c, h, w = self.output.size()

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output.view(-1, c, h, w), self.gt.view(-1, c, h, w))
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # local discriminative learning loss
            if self.cri_ldl:
                weight_map = loss_util.cal_weight_map_with_ema_result(self.gt.view(-1, c, h, w),
                                                                      self.output.view(-1, c, h, w),
                                                                      self.output_ema.view(-1, c, h, w),
                                                                      ksize=7).view(b, n, 1, h, w)
                l_g_ldl = self.cri_ldl(torch.mul(weight_map, self.output),
                                       torch.mul(weight_map, self.gt))
                l_g_total += l_g_ldl
                loss_dict['l_g_ldl'] = l_g_ldl
            # gan loss
            fake_g_pred = self.net_d(self.output.view(-1, c, h, w))
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        # reshape to (b*n, c, h, w)
        real_d_pred = self.net_d(self.gt.view(-1, c, h, w))
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        # reshape to (b*n, c, h, w)
        fake_d_pred = self.net_d(self.output.view(-1, c, h, w).detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class VideoRecurrentSTGANLDLModel(VideoRecurrentModel):

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # build network net_g with Exponential Moving Average (EMA)
            # net_g_ema only used for testing on one GPU and saving.
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        if train_opt['fix_flow']:
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:  # The fix_flow now only works for spynet.
                    flow_params.append(param)
                else:
                    normal_params.append(param)

            optim_params = [
                {  # add flow params first
                    'params': flow_params,
                    'lr': train_opt['lr_flow']
                },
                {
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
            ]
        else:
            optim_params = self.net_g.parameters()

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        if self.fix_flow_iter:
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        if self.ema_decay > 0:
            self.output_ema = self.net_g_ema(self.lq)

        b, n, c, h, w = self.output.size()

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output.view(-1, c, h, w), self.gt.view(-1, c, h, w))
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # local discriminative learning loss
            if self.cri_ldl:
                weight_map = loss_util.cal_weight_map_with_ema_result(self.gt.view(-1, c, h, w),
                                                                      self.output.view(-1, c, h, w),
                                                                      self.output_ema.view(-1, c, h, w),
                                                                      ksize=7).view(b, n, 1, h, w)
                l_g_ldl = self.cri_ldl(torch.mul(weight_map, self.output),
                                       torch.mul(weight_map, self.gt))
                l_g_total += l_g_ldl
                loss_dict['l_g_ldl'] = l_g_ldl
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        # reshape to (b*n, c, h, w)
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        # reshape to (b*n, c, h, w)
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class VideoRecurrentGANLRFlowLDLModel(VideoRecurrentModel):

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # build network net_g with Exponential Moving Average (EMA)
            # net_g_ema only used for testing on one GPU and saving.
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)

        if train_opt.get('flo_opt'):
            self.cri_flo = build_loss(train_opt['flo_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        if train_opt['fix_flow']:
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:  # The fix_flow now only works for spynet.
                    flow_params.append(param)
                else:
                    normal_params.append(param)

            optim_params = [
                {  # add flow params first
                    'params': flow_params,
                    'lr': train_opt['lr_flow']
                },
                {
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
            ]
        else:
            optim_params = self.net_g.parameters()

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        if self.fix_flow_iter:
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        self.optimizer_g.zero_grad()
        self.output, flow_f_lr, flow_b_lr = self.net_g(self.lq)
        if self.ema_decay > 0:
            self.output_ema, _, _ = self.net_g_ema(self.lq)

        b, n, c, h, w = self.output.size()

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output.view(-1, c, h, w), self.gt.view(-1, c, h, w))
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # local discriminative learning loss
            if self.cri_ldl:
                weight_map = loss_util.cal_weight_map_with_ema_result(self.gt.view(-1, c, h, w),
                                                                      self.output.view(-1, c, h, w),
                                                                      self.output_ema.view(-1, c, h, w),
                                                                      ksize=7).view(b, n, 1, h, w)
                l_g_ldl = self.cri_ldl(torch.mul(weight_map, self.output),
                                       torch.mul(weight_map, self.gt))
                l_g_total += l_g_ldl
                loss_dict['l_g_ldl'] = l_g_ldl
                # mask-based temporal consistency loss
                if self.cri_flo:
                    flow_f_hr = arch_util.resize_flow(flow_f_lr.view(-1, 2, flow_f_lr.size(3), flow_f_lr.size(4)),
                                                      'shape', [h, w], interp_mode='bilinear').detach()
                    flow_b_hr = arch_util.resize_flow(flow_b_lr.view(-1, 2, flow_b_lr.size(3), flow_b_lr.size(4)),
                                                      'shape', [h, w], interp_mode='bilinear').detach()
                    flow_f_hr = flow_f_hr.view(b, n - 1, 2, h, w)
                    flow_b_hr = flow_b_hr.view(b, n - 1, 2, h, w)
                    warped_output_next, warped_output_prev = \
                        arch_util.flow_warp_sequence_v2(self.output, flow_f_hr, flow_b_hr)
                    l_g_flo = self.cri_flo(torch.mul(weight_map[:, 1:, :, :, :], warped_output_next),
                                           torch.mul(weight_map[:, 1:, :, :, :], self.gt[:, 1:, :, :, :])) + \
                              self.cri_flo(torch.mul(weight_map[:, :-1, :, :, :], warped_output_prev),
                                           torch.mul(weight_map[:, :-1, :, :, :], self.gt[:, :-1, :, :, :]))
                    l_g_total += l_g_flo
                    loss_dict['l_g_flo'] = l_g_flo
            # gan loss
            fake_g_pred = self.net_d(self.output.view(-1, c, h, w))
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        # reshape to (b*n, c, h, w)
        real_d_pred = self.net_d(self.gt.view(-1, c, h, w))
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        # reshape to (b*n, c, h, w)
        fake_d_pred = self.net_d(self.output.view(-1, c, h, w).detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def test(self):
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            self.output, _, _ = self.net_g(self.lq)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()


@MODEL_REGISTRY.register()
class VideoRecurrentGANHRFlowLDLModel(VideoRecurrentModel):

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # build network net_g with Exponential Moving Average (EMA)
            # net_g_ema only used for testing on one GPU and saving.
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)

        if train_opt.get('flo_opt'):
            self.cri_flo = build_loss(train_opt['flo_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        if train_opt['fix_flow']:
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:  # The fix_flow now only works for spynet.
                    flow_params.append(param)
                else:
                    normal_params.append(param)

            optim_params = [
                {  # add flow params first
                    'params': flow_params,
                    'lr': train_opt['lr_flow']
                },
                {
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
            ]
        else:
            optim_params = self.net_g.parameters()

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        if self.fix_flow_iter:
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        if self.ema_decay > 0:
            self.output_ema = self.net_g_ema(self.lq)

        b, n, c, h, w = self.output.size()

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output.view(-1, c, h, w), self.gt.view(-1, c, h, w))
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # local discriminative learning loss
            if self.cri_ldl:
                weight_map = loss_util.cal_weight_map_with_ema_result(self.gt.view(-1, c, h, w),
                                                                      self.output.view(-1, c, h, w),
                                                                      self.output_ema.view(-1, c, h, w),
                                                                      ksize=7).view(b, n, 1, h, w)
                l_g_ldl = self.cri_ldl(torch.mul(weight_map, self.output),
                                       torch.mul(weight_map, self.gt))
                l_g_total += l_g_ldl
                loss_dict['l_g_ldl'] = l_g_ldl
                # mask-based temporal consistency loss
                if self.cri_flo:
                    flow_f_hr, flow_b_hr = self.get_flow(self.gt, self.get_bare_model(self.net_g).spynet)
                    flow_f_hr, flow_b_hr = flow_f_hr.detach(), flow_b_hr.detach()
                    warped_output_next, warped_output_prev = \
                        arch_util.flow_warp_sequence_v2(self.output, flow_f_hr, flow_b_hr)
                    l_g_flo = self.cri_flo(torch.mul(weight_map[:, 1:, :, :, :], warped_output_next),
                                           torch.mul(weight_map[:, 1:, :, :, :], self.gt[:, 1:, :, :, :])) + \
                              self.cri_flo(torch.mul(weight_map[:, :-1, :, :, :], warped_output_prev),
                                           torch.mul(weight_map[:, :-1, :, :, :], self.gt[:, :-1, :, :, :]))
                    l_g_total += l_g_flo
                    loss_dict['l_g_flo'] = l_g_flo
            # gan loss
            fake_g_pred = self.net_d(self.output.view(-1, c, h, w))
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        # reshape to (b*n, c, h, w)
        real_d_pred = self.net_d(self.gt.view(-1, c, h, w))
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        # reshape to (b*n, c, h, w)
        fake_d_pred = self.net_d(self.output.view(-1, c, h, w).detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def test(self):
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            self.output = self.net_g(self.lq)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()

    def get_flow(self, x, flownet):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = flownet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward
