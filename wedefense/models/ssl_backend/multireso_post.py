#!/usr/bin/env python

# Copyright 2023 National Institute of Informatics (author: Lin Zhang, zhanglin@nii.ac.jp)  # noqa
# Licensed under the BSD 3-Clause License.
'''
    implement multi-reso CM
    Title: The PartialSpoof Database and Countermeasures for the Detection of
           Short Fake Speech Segments Embedded in an Utterance
           https://ieeexplore.ieee.org/document/10003971
    Author: Lin Zhang, Xin Wang, Erica Cooper, Junichi Yamagishi, Nicholas Evans
'''

import torch
import torch.nn as nn

from wedefense.models.gmlp import GMLPBlock
from wedefense.models.pooling_layers import SelfWeightedPooling

# For SSL
from collections.abc import Iterable


def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


def set_freeze_by_names_deep(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        for name_d, child_d in child.named_children():
            if name_d not in layer_names:
                continue
            for param in child_d.parameters():
                param.requires_grad = not freeze


def freeze_by_names_deep(model, layer_names):
    set_freeze_by_names_deep(model, layer_names, True)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayer_C(SELayer):

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class SELayer_TF(nn.Module):

    def __init__(self, pool_type='avg', kernel_size=[2, 2], stride=[2, 2]):
        super(SELayer_TF, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.ex = nn.Sigmoid()
        self.pool_type = pool_type

    def forward(self, x):
        b, c, t, f = x.size()
        y = self.pool(x)
        if (self.pool_type == 'max'):
            y = torch.max(y, dim=1).values.unsqueeze(
                1)  # B x C x T x F -> B x 1 x T x F
        else:
            y = torch.mean(y, dim=1).unsqueeze(1)  # B x C x T x F
        # print(y.shape)
        y = self.ex(y)
        return y


class Branch(nn.Module):

    def __init__(self, dim, embed_dim=64, flag_pool="None", blstm_layers=0):
        """
        Input:
          dim: input_features dim of input.
          embed_dim: int. the dim of output (out_features).
            >0: the fixed vaule
            <0: the reduced multiple value, like -2 for reduce by half.
          flag_pool: for utt Branch. pooling layer used to transfer embeeding
                     from segment level to utterance level.
          blstm_layers: blstm number in postnet. >=0

        For Branch:
          Input: hidden feature: B x T x H
          Output: B x T x embed_dim for segment, or B x T for utterance.
        """
        super(Branch, self).__init__()
        self.dim = dim
        if (embed_dim > 0):
            self.embed_dim = embed_dim
        elif (embed_dim < 0):
            # if embed_dim <0, we will reduce dim by embed_dim.
            # like -2 will be dim/2
            self.embed_dim = int(dim / abs(embed_dim))

        self.flag_pool = flag_pool
        self.use_blstm = blstm_layers > 0
        if (self.flag_pool == 'sap'):
            self.pool = SelfWeightedPooling(self.dim, mean_only=True)

        if (self.use_blstm):
            layers = []
            for i in range(0, blstm_layers):
                layers.append(
                    nn.LSTM(self.dim, self.dim // 2, bidirectional=True))
            self.layers = nn.Sequential(*layers)

        self.fc = nn.Linear(self.dim, self.embed_dim, bias=False)

    def forward(self, x):

        # BLSTM layer in postnet
        if (self.use_blstm):
            x = self.layers(x) + x

        # Pooling layer in postnet, for utterance only.
        if (self.flag_pool == "ap"):  # average pooling
            x = x.mean(1)
            # nn.AdaptiveAvgPool2d(1)(x)
        elif (self.flag_pool == "sap"):
            x = self.pool(x)
        else:
            pass

        # FC in postnet.
        x = self.fc(x)
        return x


class MaxPool1d_scales(nn.Module):
    # The main modules for multi-reso structure.
    # Most modules are Inherits this class.
    def __init__(self,
                 num_scale=5,
                 feat_dim=60,
                 embed_dim=64,
                 flag_pool='ap',
                 multi_reso_active=['utt'],
                 Frame_shifts=[2, 4, 8, 16, 32, 64]):
        super(MaxPool1d_scales, self).__init__()
        self.num_scale = num_scale
        self.Frame_shifts = Frame_shifts
        assert len(multi_reso_active) > 0
        self.multi_reso_active = multi_reso_active
        active_raw = set(str(k) for k in self.multi_reso_active)
        self.active_indices = self.convert_active_reso_to_index(active_raw)

        # Set up downsampling module
        # (from (0)20ms -> (5)640ms -> (6)utt, 7 in total)
        self.blocks = nn.ModuleDict()
        self.blocks['0'] = nn.Sequential()
        for b_idx in range(1, num_scale - 1):
            self.blocks[f'{b_idx}'] = nn.Sequential(
                nn.MaxPool2d([2, 2], [2, 2]))
        self.blocks[f'{num_scale-1}'] = nn.Sequential(
            nn.MaxPool2d([2, 2], [2, 2], padding=[1, 0]), )

        self.blocks['utt'] = nn.Sequential(nn.Dropout(0.7))

        # Set up Branch (scoring modules),
        self.post_nets_seg = nn.ModuleDict()
        for i in range(num_scale):
            dim = feat_dim // pow(2, i)
            self.post_nets_seg[f"disc_{i}"] = Branch(dim=dim,
                                                     embed_dim=embed_dim,
                                                     flag_pool="None")

        # utt
        self.post_nets_utt = Branch(dim=dim,
                                    embed_dim=embed_dim,
                                    flag_pool=flag_pool)
        self.freeze_unused_para()

    def forward(self, x):
        # num_scale= 5 # the number of output
        outs = []

        for idx, (key, disc) in enumerate(self.post_nets_seg.items()):
            # print(x.shape) #check
            x = self.blocks[f"{idx}"](x)
            o = self.post_nets_seg[key](x)
            if (str(idx) in self.active_indices):
                outs.append(o)
                # outs.append(disc(x))

        in_utt = self.blocks['utt'](x)
        o_utt = self.post_nets_utt(in_utt)

        if ('utt' in self.active_indices):
            outs.append(o_utt)

        return outs[0] if len(outs) == 1 else outs

    def convert_active_reso_to_index(self, active_raw):
        # Handle multi_reso_active as either frame shifts or scale indices
        frame_shift_to_index = {
            str(fs): str(i)
            for i, fs in enumerate(self.Frame_shifts)
        }

        # Convert reso values (e.g. '4') to index keys (e.g. '1') if needed
        active_indices = set()
        for k in active_raw:
            if k == 'utt':
                active_indices.add('utt')
            elif k in frame_shift_to_index:
                active_indices.add(frame_shift_to_index[k])
            elif k.isdigit():
                active_indices.add(k)  # assume already scale index
        return active_indices

    def freeze_unused_para(self):
        """
        Freeze blocks(downsampling) and post_nets_seg that are not in self.multi_reso_active.  # noqa
        support ['2', '4', ..., 'utt'] in multi_reso_active
        """

        # Freeze block (downsampling modules)
        # Generate those require updated downsampling module(block)
        # based on their dependency:
        # e.g., scale 2 requires downsampling blocks 0 and 1.
        required_block_keys = set()
        for idx in self.active_indices:
            if idx == 'utt':
                required_block_keys.update(
                    str(i) for i in range(len(self.Frame_shifts)))
                required_block_keys.add('utt')
            elif idx.isdigit():
                required_block_keys.update(str(i) for i in range(int(idx) + 1))

        for key, module in self.blocks.items():
            # if(key == 'utt'):
            #    if('utt' not in active_indices):
            #        for param in module.parameters():
            #            param.requires_grad_(False)
            if (key not in required_block_keys):
                for param in module.parameters():
                    param.requires_grad_(False)

        # Freeze post_nets_seg:
        for key, module in self.post_nets_seg.items():
            idx_str = key.replace("disc_", "")  # e.g., "disc_1" -> "1"
            if (idx_str not in self.active_indices):
                for param in module.parameters():
                    param.requires_grad_(False)

        # Freeze utt scoring module:
        if ('utt' not in self.active_indices):
            for param in self.post_nets_utt.parameters():
                param.requires_grad_(False)

        # debug
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"[Frozen] {name}")


class SSL_BACKEND_multireso_MaxPool1d_blstmlinear(MaxPool1d_scales):

    def __init__(self,
                 num_scale=6,
                 feat_dim=60,
                 embed_dim=64,
                 blstm_layers=1,
                 flag_pool='ap',
                 multi_reso_active=['utt'],
                 Frame_shifts=[2, 4, 8, 16, 32, 64]):
        super(MaxPool1d_scales, self).__init__()

        self.blocks = MaxPool1d_scales(num_scale, feat_dim, embed_dim,
                                       flag_pool).blocks
        self.Frame_shifts = Frame_shifts
        self.multi_reso_active = multi_reso_active
        active_raw = set(str(k) for k in self.multi_reso_active)
        self.active_indices = self.convert_active_reso_to_index(active_raw)

        self.post_nets_seg = nn.ModuleDict()
        for i in range(num_scale):
            dim = feat_dim // pow(2, i)
            self.post_nets_seg[f"disc_{i}"] = Branch(dim=dim,
                                                     embed_dim=embed_dim,
                                                     flag_pool="None",
                                                     blstm_layers=blstm_layers)
        # utt
        self.post_nets_utt = Branch(dim=dim,
                                    embed_dim=embed_dim,
                                    flag_pool=flag_pool,
                                    blstm_layers=blstm_layers)
        self.freeze_unused_para()


class SSL_BACKEND_multireso_MaxPool1dLin(MaxPool1d_scales):

    def __init__(self,
                 num_scale=6,
                 feat_dim=60,
                 embed_dim=64,
                 flag_pool='ap',
                 multi_reso_active=['utt'],
                 Frame_shifts=[2, 4, 8, 16, 32, 64]):
        super(MaxPool1d_scales, self).__init__()
        self.Frame_shifts = Frame_shifts
        self.multi_reso_active = multi_reso_active
        active_raw = set(str(k) for k in self.multi_reso_active)
        self.active_indices = self.convert_active_reso_to_index(active_raw)

        self.blocks = nn.ModuleDict()
        self.blocks['0'] = nn.Sequential()
        for b_idx in range(1, num_scale - 1):
            self.blocks[f'{b_idx}'] = nn.Sequential(
                nn.MaxPool2d([2, 2], [2, 2]),
                nn.Linear(feat_dim // pow(2, b_idx),
                          feat_dim // pow(2, b_idx)))
        self.blocks[f'{num_scale-1}'] = nn.Sequential(
            nn.MaxPool2d([2, 2], [2, 2], padding=[1, 0]),
            nn.Linear(feat_dim // pow(2, num_scale - 1),
                      feat_dim // pow(2, num_scale - 1)),
        )
        self.blocks['utt'] = nn.Sequential(nn.Dropout(0.7))

        self.post_nets_seg = nn.ModuleDict()
        for i in range(num_scale):
            dim = feat_dim // pow(2, i)
            self.post_nets_seg[f"disc_{i}"] = Branch(dim, embed_dim, "None")
        # utt
        self.post_nets_utt = Branch(dim, embed_dim, flag_pool)
        self.freeze_unused_para()


class SSL_BACKEND_multireso_MaxPool1dLin_blstmlinear(
        SSL_BACKEND_multireso_MaxPool1d_blstmlinear):

    def __init__(self,
                 num_scale=6,
                 feat_dim=60,
                 embed_dim=64,
                 blstm_layers=1,
                 flag_pool='ap',
                 multi_reso_active=['utt'],
                 Frame_shifts=[2, 4, 8, 16, 32, 64]):
        super(SSL_BACKEND_multireso_MaxPool1d_blstmlinear, self).__init__()
        self.Frame_shifts = Frame_shifts
        self.multi_reso_active = multi_reso_active
        active_raw = set(str(k) for k in self.multi_reso_active)
        self.active_indices = self.convert_active_reso_to_index(active_raw)

        self.blocks = SSL_BACKEND_multireso_MaxPool1dLin(
            num_scale, feat_dim, embed_dim, flag_pool).blocks

        base_model = SSL_BACKEND_multireso_MaxPool1d_blstmlinear(
            num_scale, feat_dim, embed_dim, blstm_layers, flag_pool)
        self.post_nets_seg = base_model.post_nets_seg

        self.post_nets_utt = base_model.post_nets_utt
        self.freeze_unused_para()


# Gmlp
class gMLP(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_ffn: int,
                 seq_len: int,
                 gmlp_layers=1,
                 batch_first=True,
                 flag_pool='None'):
        """
        gmlp_layer: number of gmlp layers
        d_model: dim(d) of input [n * d]
        d_ffn: dim of hidden feature
        seq_len: the max of input n. for mask
        batch_first:

        """
        super().__init__()

        self.batch_first = batch_first
        self.flag_pool = flag_pool

        if (d_ffn > 0):
            pass
        elif (d_ffn < 0):
            # if embed_dim <0, we will reduce dim by embed_dim.
            # like -2 will be dim/2
            d_ffn = int(d_model / abs(d_ffn))

        layers = []
        for i in range(gmlp_layers):
            layers.append(GMLPBlock(d_model, d_ffn, seq_len))
        self.layers = nn.Sequential(*layers)

        if (self.flag_pool == 'sap'):
            self.pool = SelfWeightedPooling(self.dim, mean_only=True)

        self.fc = nn.Linear(d_model, d_ffn, bias=False)

    def forward(self, x):

        if (self.batch_first):
            x = x.permute(1, 0, 2)
            x = self.layers(x)
            x = x.permute(1, 0, 2)
        else:
            x = self.layers(x)

        # pool for utt
        if (self.flag_pool == "ap"):  # average pooling
            x = x.mean(1)
            # nn.AdaptiveAvgPool2d(1)(x)
        elif (self.flag_pool == "sap"):
            x = self.pool(x)
        else:
            pass

        x = self.fc(x)

        return x


class SSL_BACKEND_multireso_MaxPool1d_gmlp(MaxPool1d_scales):
    # Inherits MaxPool1d_scales
    def __init__(self,
                 num_scale=5,
                 feat_dim=60,
                 embed_dim=256,
                 seq_len=2001,
                 gmlp_layers=1,
                 batch_first=True,
                 flag_pool='ap',
                 multi_reso_active=[''],
                 Frame_shifts=[2, 4, 8, 16, 32, 64]):

        super(MaxPool1d_scales, self).__init__()

        self.Frame_shifts = Frame_shifts
        self.multi_reso_active = multi_reso_active
        active_raw = set(str(k) for k in self.multi_reso_active)
        self.active_indices = self.convert_active_reso_to_index(active_raw)

        self.blocks = MaxPool1d_scales(num_scale, feat_dim, embed_dim,
                                       flag_pool).blocks

        self.post_nets_seg = nn.ModuleDict()
        for i in range(num_scale):
            dim = feat_dim // pow(2, i)
            self.post_nets_seg[f"disc_{i}"] = gMLP(dim,
                                                   embed_dim,
                                                   seq_len // pow(2, i),
                                                   gmlp_layers=gmlp_layers,
                                                   batch_first=batch_first,
                                                   flag_pool='None')
        # utt
        self.post_nets_utt = gMLP(dim,
                                  embed_dim,
                                  seq_len // pow(2, i),
                                  gmlp_layers=gmlp_layers,
                                  batch_first=batch_first,
                                  flag_pool=flag_pool)
        self.freeze_unused_para()


class SSL_BACKEND_multireso_MaxPool1dLin_gmlp(MaxPool1d_scales):

    def __init__(self,
                 num_scale=5,
                 feat_dim=60,
                 embed_dim=256,
                 seq_len=2001,
                 gmlp_layers=1,
                 batch_first=True,
                 flag_pool='ap',
                 multi_reso_active=[''],
                 Frame_shifts=[2, 4, 8, 16, 32, 64]):

        super(MaxPool1d_scales, self).__init__()
        self.Frame_shifts = Frame_shifts
        self.multi_reso_active = multi_reso_active
        active_raw = set(str(k) for k in self.multi_reso_active)
        self.active_indices = self.convert_active_reso_to_index(active_raw)

        self.blocks = SSL_BACKEND_multireso_MaxPool1dLin(
            num_scale,
            feat_dim,
            embed_dim,
            flag_pool,
            Frame_shifts=Frame_shifts).blocks

        self.post_nets_seg = nn.ModuleDict()
        for i in range(num_scale):
            dim = feat_dim // pow(2, i)
            self.post_nets_seg[f"disc_{i}"] = gMLP(dim,
                                                   embed_dim,
                                                   seq_len // pow(2, i),
                                                   gmlp_layers=gmlp_layers,
                                                   batch_first=batch_first,
                                                   flag_pool='None')
        # utt
        self.post_nets_utt = gMLP(dim,
                                  embed_dim,
                                  seq_len // pow(2, i),
                                  gmlp_layers=gmlp_layers,
                                  batch_first=batch_first,
                                  flag_pool=flag_pool)
        self.freeze_unused_para()


def debug():
    t = torch.rand((8, 48, 768))
    # model=SSL_BACKEND_multireso_MaxPool1dLin(
    #    num_scale=7, feat_dim=768, flag_pool='ap')
    model = SSL_BACKEND_multireso_MaxPool1d_gmlp(num_scale=6,
                                                 feat_dim=768,
                                                 embed_dim=-2,
                                                 seq_len=2001,
                                                 gmlp_layers=1,
                                                 batch_first=True,
                                                 flag_pool='ap')
    model_ml = SSL_BACKEND_multireso_MaxPool1dLin_gmlp(num_scale=6,
                                                       feat_dim=768,
                                                       embed_dim=-2,
                                                       seq_len=2001,
                                                       gmlp_layers=1,
                                                       batch_first=True,
                                                       flag_pool='ap')
    o = model(t)
    for i in o:  # (B x T x F) 20 ms -> 640ms -> utt.
        print(i.shape)


if __name__ == '__main__':
    debug()
