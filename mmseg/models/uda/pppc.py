# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy
from .proto import Prototypes
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks, get_mean_std, strong_transform,renorm01, denorm01)
from mmseg.models.utils.visualization import subplotimg
from mmseg.models.utils.ours_transforms import RandomCrop, RandomCropNoProd
from mmseg.models.utils.proto_estimator import ProtoEstimator
from mmseg.models.losses.contrastive_loss import contrast_preparations


class RampdownScheduler(object):

    def __init__(self, begin_iter, max_iter, current_iter, max_value, min_value, ramp_mult, step_size):
        super().__init__()
        self.begin_iter = int(begin_iter)
        self.max_iter = int(max_iter)
        self.max_value = float(max_value)
        self.mult = float(ramp_mult)
        self.iter = current_iter
        self.min_value = min_value
        self.step_size = step_size

    def step(self):
        self.iter += self.step_size

    @property
    def value(self):
        current_value = self.get_lr(self.iter, self.begin_iter, self.max_iter, self.max_value, self.min_value,
                                    self.mult)
        if current_value < self.min_value:
            current_value = self.min_value
        return current_value

    @staticmethod
    def get_lr(iter, begin_iter, max_iters, max_val, min_value, mult):
        if iter < begin_iter:
            return 0.
        elif iter >= max_iters:
            return min_value
        return max_val * np.exp(mult * (float(iter - begin_iter) / (max_iters - begin_iter)) ** 2)


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val




def calc_thresh_matrix(ema_target_softmax, var, log_vars, iter):
    var_max, _ = torch.max(var, dim=1)
    var2prob = 0.6 + (iter / 40000) * 0.2 + (var_max / (torch.max(var_max) + 1e-3) * 0.2)
    # thresh = 0.968
    pseudo_prob, pseudo_label = torch.max(ema_target_softmax, dim=1)
    thresh_matrix = torch.zeros(pseudo_label.shape).cuda()
    for i in range(19):
        class_mask = (pseudo_label==i)
        if torch.sum(class_mask) == 0:
            continue
        proto_thresh = var2prob[i]
        thresh_matrix[class_mask] = (pseudo_prob[class_mask] > proto_thresh).float()
        percentage = torch.sum(pseudo_prob[class_mask] > proto_thresh) / torch.sum(pseudo_prob[class_mask] > -1)
        log_vars[f'class {i} percentage'] = percentage.cpu().numpy()
    # ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
    return thresh_matrix


@UDA.register_module()
class PPPC(UDADecorator):

    def __init__(self, **cfg):
        super(PPPC, self).__init__(**cfg)
        # basic setup
        self.local_iter = 0
        # self.sche_sch = RampdownScheduler(0, 40000, 0, 1.0, 0, -5.0, 1)
        self.sche_sch = RampdownScheduler(0, 40000, 0, 1.0, 0, -1.0, 1)
        self.threshold_strong = 0.5
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']

        # for ssl
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        assert self.mix == 'class'
        self.enable_self_training = cfg['enable_self_training']
        self.enable_strong_aug = cfg['enable_strong_aug']
        self.push_off_self_training = cfg.get('push_off_self_training', False)

        # configs for contrastive
        self.proj_dim = cfg['model']['auxiliary_head']['channels']
        self.contrast_mode = cfg['model']['auxiliary_head']['input_transform']
        self.calc_layers = cfg['model']['auxiliary_head']['in_index']
        self.num_classes = cfg['model']['decode_head']['num_classes']
        # self.enable_avg_pool = cfg['model']['auxiliary_head']['loss_decode']['use_avg_pool']
        # self.scale_min_ratio = cfg['model']['auxiliary_head']['loss_decode']['scale_min_ratio']

        # iter to start cl
        self.start_distribution_iter = cfg['start_distribution_iter']

        # for prod strategy (CBC)
        self.pseudo_random_crop = cfg.get('pseudo_random_crop', False)
        self.crop_size = cfg.get('crop_size', (640, 640))
        self.cat_max_ratio = cfg.get('cat_max_ratio', 0.75)
        self.regen_pseudo = cfg.get('regen_pseudo', False)
        self.prod = cfg.get('prod', True)

        # feature storage for contrastive
        self.feat_distributions = None
        self.ignore_index = 255

        # BankCL memory length
        # self.memory_length = cfg.get('memory_length', 0)  # 0 means no memory bank
        self.tgt_proto = Prototypes()

        
        # ema model
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        a = self.get_ema_model().named_parameters()
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def random_crop(self, image, gt_seg, true_gt=None, prod=True, proto_sigma=None, dataset=None):
        if prod:
            RC = RandomCrop(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        else:
            RC = RandomCropNoProd(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        assert self.pseudo_random_crop
        image = image.permute(0, 2, 3, 1).contiguous()
        gt_seg = gt_seg
        true_gt = true_gt.squeeze(1)
        res_img, res_gt, res_true_gt, crop_box = [], [], [], []
        for img, gt, true_gt in zip(image, gt_seg, true_gt):
            results = {'img': img, 'gt_semantic_seg': gt, 'seg_fields': ['gt_semantic_seg'], 'true_gt': true_gt,
                       'proto_sigma': proto_sigma, 'dataset': dataset}
            results = RC(results)
            img, gt, true_gt = results['img'], results['gt_semantic_seg'], results['true_gt']
            res_img.append(img.unsqueeze(0))
            res_gt.append(gt.unsqueeze(0))
            res_true_gt.append(true_gt.unsqueeze(0))
            crop_box.append(results['crop_box'])
        image = torch.cat(res_img, dim=0).permute(0, 3, 1, 2).contiguous()
        gt_seg = torch.cat(res_gt, dim=0).long()
        gt_true_seg = torch.cat(res_true_gt, dim=0).long()
        return image, gt_seg, gt_true_seg, crop_box

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img, target_img_metas, target_gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        # torch.autograd.set_detect_anomaly(True)
        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        weak_img, weak_target_img = img.clone(), target_img.clone()

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        # Generate pseudo-label
        ema_target_logits = self.get_ema_model().encode_decode(weak_target_img, target_img_metas)
        ema_target_softmax = torch.softmax(ema_target_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_target_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size

        proto_sigma = torch.mean(self.tgt_proto.get_proto()[1], dim=1)
        if self.pseudo_random_crop:
            weak_target_img, pseudo_label, gt_true_seg, crop_box = self.random_crop(weak_target_img, pseudo_label,
                                                                                    true_gt=target_gt_semantic_seg,
                                                                                    prod=self.prod,
                                                                                    proto_sigma=proto_sigma,
                                                                                    dataset='gta')
            if self.regen_pseudo:
                # Re-Generate pseudo-label
                ema_target_logits = self.get_ema_model().encode_decode(weak_target_img, target_img_metas)
                # target_raw_feature = self.get_ema_model().extract_seg_logit(weak_target_img, target_img_metas)
                ema_target_softmax = torch.softmax(ema_target_logits.detach(), dim=1)
                pseudo_prob, pseudo_label = torch.max(ema_target_softmax, dim=1)
                # pseudo_weight = calc_thresh_matrix(ema_target_softmax, self.tgt_proto.get_proto()[1], log_vars, self.local_iter)
                ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
                ps_size = np.size(np.array(pseudo_label.cpu()))
                pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            target_img = weak_target_img.clone()




        img, gt_semantic_seg = strong_transform(
            strong_parameters,
            data=weak_img,
            target=gt_semantic_seg
        )
        target_img, _ = strong_transform(
            strong_parameters,
            data=target_img,
            target=pseudo_label.unsqueeze(1)
        )

        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_label.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)


        with torch.no_grad():
            weak_gt_semantic_seg = gt_semantic_seg.clone().detach()
            # update distribution
            ema_src_feat = self.get_ema_model().extract_auxiliary_feat(weak_img)
            self.tgt_proto.update_prototypes(ema_src_feat, weak_gt_semantic_seg,
                                             None)

        sche = self.sche_sch
        # # if self.local_iter % 200 == 0:
        # #     sche.step()
        sche.step()
        # sche_value = sche.value
        log_vars["sche_value"] = sche.value


        # source ce + cl
        src_mode = 'dec'  # stands for ce only
        if self.local_iter >= self.start_distribution_iter:
            src_mode = 'all'  # stands for ce + cl

        source_losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=False,
                                                       mode=src_mode, bank=self.tgt_proto, domain='src',
                                                       scale=sche.value, log_vars=log_vars)  # 两张source
        source_loss, source_log_vars = self._parse_losses(source_losses)
        log_vars.update(add_prefix(source_log_vars, 'src'))
        # source_loss.backward()
        # for name, parms in self.model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
        #           ' -->grad_value:', torch.mean(parms.grad))
        # try:
        source_loss.backward()
        # except:
        #     for name, parms in self.model.named_parameters():
        #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
        #               ' -->grad_value:', torch.mean(parms.grad))

        if self.local_iter >= self.start_distribution_iter:
            # target cl
            pseudo_lbl = pseudo_label.clone()  # pseudo label should not be overwritten
            pseudo_lbl[pseudo_weight == 0.] = self.ignore_index
            pseudo_lbl = pseudo_lbl.unsqueeze(1)
            target_losses = self.get_model().forward_train(target_img, target_img_metas, pseudo_lbl, return_feat=False,
                                                           mode='aux',
                                                           bank=self.tgt_proto, domain='tgt', scale=sche.value,
                                                           log_vars=log_vars)
            target_loss, target_log_vars = self._parse_losses(target_losses)
            log_vars.update(add_prefix(target_log_vars, 'tgt'))
            target_loss.backward()

        local_enable_self_training = \
            self.enable_self_training and \
            (not self.push_off_self_training or self.local_iter >= self.start_distribution_iter)

        # mixed ce (ssl)
        if local_enable_self_training:
            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mix_masks = get_class_masks(gt_semantic_seg)  #

            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((weak_img[i], weak_target_img[i])),
                    target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            mixed_img = torch.cat(mixed_img)


            mixed_lbl = torch.cat(mixed_lbl)
            mixed_mask = torch.cat(mix_masks)
            # Train on mixed images
            mix_losses = self.get_model().forward_train(mixed_img, img_metas, mixed_lbl, pseudo_weight,
                                                        return_feat=False, mode='dec', bank=self.tgt_proto,
                                                        domain='mix')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(add_prefix(mix_log_vars, 'mix'))
            mix_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'visualize_meta')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            if local_enable_self_training:
                vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            ema_src_logits = self.get_ema_model().encode_decode(weak_img, img_metas)
            ema_softmax = torch.softmax(ema_src_logits.detach(), dim=1)
            _, src_pseudo_label = torch.max(ema_softmax, dim=1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], f'{img_metas[j]["ori_filename"]}')
                subplotimg(axs[1][0], vis_trg_img[j],
                           f'{os.path.basename(target_img_metas[j]["ori_filename"]).replace("_leftImg8bit", "")}')
                subplotimg(
                    axs[0][1],
                    src_pseudo_label[j],
                    'Source Pseudo Label',
                    cmap='cityscapes',
                    nc=self.num_classes)
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Pseudo Label',
                    cmap='cityscapes',
                    nc=self.num_classes)
                subplotimg(
                    axs[0][2],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes',
                    nc=self.num_classes)
                if target_gt_semantic_seg.dim() > 1:
                    subplotimg(
                        axs[1][2],
                        target_gt_semantic_seg[j],
                        'Target Seg GT',
                        cmap='cityscapes',
                        nc=self.num_classes
                    )
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if local_enable_self_training:
                    subplotimg(
                        axs[1][3],
                        mix_masks[j][0],
                        'Mixed Mask',
                        cmap='gray'
                    )
                    subplotimg(
                        axs[0][4],
                        vis_mixed_img[j],
                        'Mixed ST Image')
                    subplotimg(
                        axs[1][4],
                        mixed_lbl[j],
                        'Mixed ST Label',
                        cmap='cityscapes',
                        nc=self.num_classes
                    )
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars
