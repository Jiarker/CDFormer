import torch
import torch.nn as nn
from logging import Logger
from mmcv import Config
import numpy as np

from .base_model import Base_model
from .builder import MODELS
from .common.utils import up_sample, dwt2d, data_normalize, data_denormalize
from .common.modules import ResBlock, conv3x3, SwinModule, conv1x1

class Transformer(nn.Module):
    def __init__(self, self_attention=True, input_chans=4, n_feats=32, n_blocks=1, downscaling_factor=1, n_heads=4, head_dim=16, win_size=4):
        super(Transformer, self).__init__()

        if self_attention:
            self.attention = SwinModule(in_channels=input_chans, hidden_dimension=n_feats, layers=n_blocks,
                                                  downscaling_factor=downscaling_factor, num_heads=n_heads, head_dim=head_dim,
                                                  window_size=win_size, relative_pos_embedding=True, cross_attn=False)
        else:
            self.attention = SwinModule(in_channels=input_chans, hidden_dimension=n_feats, layers=n_blocks,
                                                  downscaling_factor=downscaling_factor, num_heads=n_heads, head_dim=head_dim,
                                                  window_size=win_size, relative_pos_embedding=True, cross_attn=False)
    def forward(self, KV, Q=None):
        if Q == None:   # 自注意力机制
            return self.attention(KV)
        else:   # 注意力机制
            return self.attention(KV, Q)


class RestoreNet(nn.Module):
    def __init__(self, logger: Logger, ms_chans, n_feats=32, norm_type='BN', basic_block=ResBlock):
            super(RestoreNet, self).__init__()

            # 图像恢复
            self.HR_tail = nn.Sequential(
                # conv3x3(n_feats, n_feats * 4),
                # # nn.PixelShuffle:用于对输入张量进行像素重排(上采样)
                # nn.PixelShuffle(2),
                # nn.ReLU(True),

                conv3x3(n_feats * 3, n_feats * 4),
                nn.PixelShuffle(2),
                nn.ReLU(True),
                conv3x3(n_feats, ms_chans))

    def forward(self, x):
        # Batch_size x 4 x 256 x 256
        x = self.HR_tail(x)
        return x

class Branch(nn.Module):
    def __init__(self, n_feats=32, n_block=1, h_attn=True, inn_iter=1):
        super(Branch, self).__init__()
        self.n_block = n_block
        self.attn = nn.ModuleList()
        self.conv = nn.ModuleList()
        for _ in range(self.n_block):
            self.attn.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, cross_attn=True, h_attn=h_attn, inn_iter=inn_iter))
        for _ in range(self.n_block-1):
            self.conv.append(conv3x3(n_feats, n_feats))
    def forward(self, x, h):
        if h == None:
            return x
        x = self.attn[0](x, h)
        h_conv = h
        for i in range(1, self.n_block):
            h_conv = self.conv[i-1](h_conv)
            nn.ReLU(True),
            x = self.attn[i](x, h_conv)
        result = x + h
        return result

class Core_Module(nn.Module):
    def __init__(self, logger, ms_chans=4, n_feats=32, norm_input=True, bit_depth=10, norm_type='BN', n_blocks=(1, 1, 3), inn_iters=(2,1)):
        super(Core_Module, self).__init__()

        self.norm_input = norm_input
        self.bit_depth = bit_depth

        self.n_blocks = n_blocks

        self.ms_encoder = SwinModule(in_channels=ms_chans*4, hidden_dimension=n_feats, cross_attn=False)
        if n_blocks[1] != 0:
            self.spe_ms_encoder = SwinModule(in_channels=ms_chans, hidden_dimension=n_feats, cross_attn=False)
        self.pan_encoder = SwinModule(in_channels=4, hidden_dimension=n_feats, downscaling_factor=1, cross_attn=False)
        if n_blocks[2] != 0:
            self.spe_pan_encoder = SwinModule(in_channels=3, hidden_dimension=n_feats, downscaling_factor=1, cross_attn=False)

        self.ms_KV_cross_attn = nn.ModuleList()
        self.pan_KV_cross_attn = nn.ModuleList()

        self.conv1 = conv1x1(n_feats*2, n_feats)
        self.conv2 = conv1x1(n_feats*3, n_feats)

        for _ in range(n_blocks[0]):
            self.ms_KV_cross_attn.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, cross_attn=True, h_attn=False, inn_iter=inn_iters[0]))
            self.pan_KV_cross_attn.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, cross_attn=True, h_attn=False, inn_iter=inn_iters[0]))

        self.l_ms_attn = Branch(n_feats=n_feats, n_block=n_blocks[1], h_attn=True, inn_iter=inn_iters[1])
        self.h_pan_attn = Branch(n_feats=n_feats, n_block=n_blocks[2], h_attn=True, inn_iter=inn_iters[1])

        self.ResBlock = RestoreNet(logger, ms_chans, n_feats, norm_type)

    def forward(self, x_ms, x_pan):

        # 输入归一化
        if self.norm_input:
            x_ms = data_normalize(x_ms, self.bit_depth)
            x_pan = data_normalize(x_pan, self.bit_depth)

        # 进行DWT变换
        l_dwt_pan, h_dwt_pan = dwt2d(x_pan)
        dwt_pan = torch.cat([l_dwt_pan, h_dwt_pan], dim=1)
        up_ms = up_sample(x_ms, r=4)
        l_dwt_ms, h_dwt_ms = dwt2d(up_ms)
        dwt_ms = torch.cat([l_dwt_ms, h_dwt_ms], dim=1)

        # 编码
        encode_ms = self.ms_encoder(dwt_ms)
        encode_spe_ms = None
        encode_spe_pan = None
        if self.n_blocks[1] != 0:
            encode_spe_ms = self.spe_ms_encoder(l_dwt_ms)

        encode_pan = self.pan_encoder(dwt_pan)
        if self.n_blocks[2] != 0:
            encode_spe_pan = self.spe_pan_encoder(h_dwt_pan)

        # 进行交叉注意力(特征融合)
        last_pan_feat = encode_pan
        last_ms_feat = encode_ms
        for i in range(self.n_blocks[0]):
            ms_cross_pan_feat = self.ms_KV_cross_attn[i](last_ms_feat, last_pan_feat)
            pan_cross_ms_feat = self.pan_KV_cross_attn[i](last_pan_feat, last_ms_feat)
            last_ms_feat = ms_cross_pan_feat
            last_pan_feat = pan_cross_ms_feat

        # 通道合并
        merge = torch.cat([last_ms_feat, last_pan_feat], dim=1)
        merge = self.conv1(merge)

        # 高频信息提取
        merge1 = self.l_ms_attn(merge, encode_spe_ms)
        merge2 = self.h_pan_attn(merge, encode_spe_pan)
        merge3 = torch.cat([merge, merge1, merge2], dim=1)
        # merge3 = self.conv2(merge3)

        # 图像恢复
        result = self.ResBlock(merge3)

        # 复原
        if self.norm_input:
            data_denormalize(result, self.bit_depth)

        return result, l_dwt_ms, h_dwt_pan

@MODELS.register_module()
class CDFormer(Base_model):
    def __init__(self, cfg, logger, train_data_loader, test_data_loader0, test_data_loader1):
        super(CDFormer, self).__init__(cfg, logger, train_data_loader, test_data_loader0, test_data_loader1)

        ms_chans = cfg.get('ms_chans', 4)

        model_cfg = cfg.get('model_cfg', dict())
        n_feats = model_cfg.get('n_feats', 32)

        Core_cfg = model_cfg.get('core_module', dict())
        self.add_module('core_module', Core_Module(logger=logger, ms_chans=ms_chans, n_feats=n_feats, **Core_cfg))

        self.to_pan_mode = model_cfg.get('to_pan_mode', 'max')

    def train_iter(self, iter_id, input_batch, log_freq=100):

        # 初始化各模型
        Core = self.module_dict['core_module']
        Core_optim = self.optim_dict['core_module']

        loss = 0.0
        loss_res = dict()
        loss_cfg = self.cfg.get('loss_cfg', {})

        # 模型运行
        # 初始化各类型的遥感图像
        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']

        # 模型运行
        output, l_dwt_ms, h_dwt_pan = Core(input_lr, input_pan)

        # 计算损失函数
        if 'QNR_loss' in self.loss_module and loss_cfg['QNR_loss'].w != 0.0 :
            input_pan_l = input_batch['input_pan_l']
            QNR_loss = self.loss_module['QNR_loss'](pan=input_pan, ms=input_lr, pan_l=input_pan_l, out=output)
            loss += QNR_loss * loss_cfg['QNR_loss'].w
            loss_res['QNR_loss'] = QNR_loss.item()

        l_out, h_out = dwt2d(output)
        target = input_batch['target']
        l_gt, h_gt = dwt2d(target)

        if 'spectral_rec_loss' in self.loss_module and loss_cfg['spectral_rec_loss'].w != 0.0:
            spectral_rec_loss = self.loss_module['spectral_rec_loss'](
                out=l_out, gt=l_gt)
            loss += spectral_rec_loss * loss_cfg['spectral_rec_loss'].w
            loss_res['spectral_rec_loss'] = spectral_rec_loss.item()

        if 'spatial_rec_loss' in self.loss_module and loss_cfg['spatial_rec_loss'].w != 0.0:
            spatial_rec_loss = self.loss_module['spatial_rec_loss'](
                out=h_out, gt=h_gt)
            loss += spatial_rec_loss * loss_cfg['spatial_rec_loss'].w
            loss_res['spatial_rec_loss'] = spatial_rec_loss.item()

        if 'rec_loss' in self.loss_module and loss_cfg['rec_loss'].w != 0.0:
            rec_loss = self.loss_module['rec_loss'](
                out=output, gt=target)
            loss += rec_loss * loss_cfg['rec_loss'].w
            loss_res['rec_loss'] = rec_loss.item()

        loss_res['full_loss'] = loss.item()

        # 更新参数
        Core_optim.zero_grad()
        loss.backward()
        Core_optim.step()

        self.print_train_log(iter_id, loss_res, log_freq)

        return loss_res