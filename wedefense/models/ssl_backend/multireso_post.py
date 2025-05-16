#!/usr/bin/env python

# Copyright 2023 National Institute of Informatics (author: Lin Zhang, zhanglin@nii.ac.jp)
# Licensed under the BSD 3-Clause License.

'''
    implement multi-reso CM
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from wedefense.models.gmlp import GMLPBlock
from wedefense.models.pooling_layers import SelfWeightedPooling

#For SSL
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
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

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
        self.ex=nn.Sigmoid()
        self.pool_type = pool_type

    def forward(self, x):
        b, c, t, f = x.size()
        y = self.pool(x)
        if(self.pool_type == 'max'):
            y = torch.max(y, dim=1).values.unsqueeze(1)  # B x C x T x F -> B x 1 x T x F
        else:
            y = torch.mean(y, dim=1).unsqueeze(1)  # B x C x T x F
        #print(y.shape)
        y = self.ex(y)
        return  y


class Branch(nn.Module):
    def __init__(self, dim, emb_dim=64, flag_pool="None", blstm_layers=0):
        """
        Input: 
          dim: input_features dim of input.
          emb_dim: int. the dim of output (out_features).
            >0: the fixed vaule
            <0: the reduced multiple value, like -2 for reduce by half. 
          flag_pool: for utt Branch. pooling layer used to transfer embeeding from segment level to utterance level.
          blstm_layers: blstm number in postnet. >=0

        For Branch:
          Input: hidden feature: B x T x H 
          Output: B x T x emb_dim for segment, or B x T for utterance.
        """
        super(Branch, self).__init__()
        self.dim = dim
        if(emb_dim > 0):
            self.emb_dim = emb_dim
        elif(emb_dim < 0):
            #if emb_dim <0, we will reduce dim by emb_dim. like -2 will be dim/2
            self.emb_dim = int(dim / abs(emb_dim))

        self.flag_pool = flag_pool
        self.use_blstm = blstm_layers > 0
        if(self.flag_pool=='sap'):
            self.pool=SelfWeightedPooling(self.dim, mean_only=True)

        if(self.use_blstm):
            layers = []
            for i in range(0, blstm_layers):
                layers.append(nn.LSTM(self.dim, self.dim // 2, \
                         bidirectional=True))
            self.layers = nn.Sequential(*layers)

        self.fc = nn.Linear(self.dim, self.emb_dim, bias=False)


    def forward(self, x):

        #BLSTM layer in postnet
        if(self.use_blstm):
            x = self.layers(x) + x

        #Pooling layer in postnet, for utterance only.    
        if(self.flag_pool == "ap"): #average pooling
            x = x.mean(1)
            #nn.AdaptiveAvgPool2d(1)(x)
        elif(self.flag_pool == "sap"):
            x = self.pool(x)
        else:
            pass

        #FC in postnet.
        x = self.fc(x)    
        return x


class MaxPool1d_scales(nn.Module):
    def __init__(self, num_scale=5, feature_F_dim=60, emb_dim = 64,flag_pool = 'ap', 
            multi_branch_fix = '', Frame_shifts = [2,4,8,16,32,64] ):
        super(MaxPool1d_scales, self).__init__()

        self.multi_branch_fix = multi_branch_fix
        self.Frame_shifts = Frame_shifts
        
        self.blocks=nn.ModuleDict()
        self.blocks['0']= nn.Sequential(
                )
        for b_idx in range(1, num_scale-1):
            self.blocks[f'{b_idx}']= nn.Sequential(
                nn.MaxPool2d([2, 2], [2, 2])
                    )
        self.blocks[f'{num_scale-1}']= nn.Sequential(
            nn.MaxPool2d([2, 2], [2, 2], padding=[1,0]),
                )

        self.blocks['utt'] = nn.Sequential(
            nn.Dropout(0.7)
            )

        self.post_nets_seg = nn.ModuleDict()
        for i in range(num_scale):
            dim = feature_F_dim // pow(2,i) 
            self.post_nets_seg[f"disc_{i}"] = Branch(dim=dim, emb_dim=emb_dim, flag_pool="None")

            if(str(self.Frame_shifts[i]) in self.multi_branch_fix):
                freeze_by_names_deep(self, f"disc_{i}")
            #else:
            #    #unfreeze_by_names(self, f"post_nets_seg.disc_{i}")
                #self.post_nets_seg[f"disc_{i}"].requires_grad = False 
        #utt    
        if('utt' in self.multi_branch_fix):
            freeze_by_names(self, "post_nets_utt")

        self.post_nets_utt = Branch(dim=dim, emb_dim=emb_dim, flag_pool=flag_pool)


    def forward(self, x):
        #num_scale= 5 # the number of output
        outs = []

        for idx, (key, disc) in enumerate(self.post_nets_seg.items()):
            #print(x.shape) #check
            x=self.blocks[f"{idx}"](x)
            o = self.post_nets_seg[key](x)
            outs.append(o)
            #outs.append(disc(x))

        in_utt = self.blocks['utt'](x)
        o_utt = self.post_nets_utt(in_utt)

        outs.append(o_utt)  

        return outs

class MaxPool1d_blstmlinear_scales(MaxPool1d_scales):
    def __init__(self, num_scale=5, feature_F_dim=60, emb_dim = 64, blstm_layers=1, flag_pool = 'ap', 
            multi_branch_fix = '', Frame_shifts = [2,4,8,16,32,64] ):
        super(MaxPool1d_scales, self).__init__()

        self.blocks=MaxPool1d_scales(num_scale, feature_F_dim, emb_dim, flag_pool).blocks 

        self.multi_branch_fix = multi_branch_fix
        self.Frame_shifts = Frame_shifts

        self.post_nets_seg = nn.ModuleDict()
        for i in range(num_scale):
            dim = feature_F_dim // pow(2,i) 
            self.post_nets_seg[f"disc_{i}"] = Branch(dim = dim, emb_dim = emb_dim, 
                    flag_pool = "None", blstm_layers = blstm_layers)
            if(str(self.Frame_shifts[i]) in self.multi_branch_fix):
                freeze_by_names_deep(self, f"disc_{i}")
        #utt    
        self.post_nets_utt = Branch(dim = dim, emb_dim = emb_dim, 
                flag_pool = flag_pool, blstm_layers = blstm_layers)
        if('utt' in self.multi_branch_fix):
            freeze_by_names(self, "post_nets_utt")


class MaxPool1dLin_scales(MaxPool1d_scales):
    def __init__(self, num_scale=5, feature_F_dim=60, emb_dim = 64,flag_pool = 'ap',
            multi_branch_fix = '', Frame_shifts = [2,4,8,16,32,64] ):
        super(MaxPool1d_scales, self).__init__()
        self.multi_branch_fix = multi_branch_fix
        self.Frame_shifts = Frame_shifts

        self.blocks=nn.ModuleDict()
        self.blocks['0']= nn.Sequential(
                )
        for b_idx in range(1, num_scale-1):
            self.blocks[f'{b_idx}']= nn.Sequential(
                nn.MaxPool2d([2, 2], [2, 2]),
                nn.Linear(feature_F_dim // pow(2, b_idx), feature_F_dim // pow(2, b_idx))
                    )
        self.blocks[f'{num_scale-1}']= nn.Sequential(
            nn.MaxPool2d([2, 2], [2, 2], padding=[1, 0]),
            nn.Linear(feature_F_dim // pow(2, num_scale-1), feature_F_dim // pow(2, num_scale-1)),

            )
        self.blocks['utt'] = nn.Sequential(
            nn.Dropout(0.7)
            )

        self.post_nets_seg = nn.ModuleDict()
        for i in range(num_scale):
            dim = feature_F_dim // pow(2,i) 
            self.post_nets_seg[f"disc_{i}"] = Branch(dim, emb_dim, "None")
            if(str(self.Frame_shifts[i]) in self.multi_branch_fix):
                freeze_by_names_deep(self, f"disc_{i}")
        #utt    
        self.post_nets_utt = Branch(dim, emb_dim, flag_pool)
        if('utt' in self.multi_branch_fix):
            freeze_by_names(self, "post_nets_utt")

class MaxPool1dLin_blstmlinear_scales(MaxPool1d_blstmlinear_scales):
    def __init__(self, num_scale=5, feature_F_dim=60, emb_dim = 64, blstm_layers=1, flag_pool = 'ap',
            multi_branch_fix = '', Frame_shifts = [2,4,8,16,32,64] ):
        super(MaxPool1d_blstmlinear_scales, self).__init__()
        self.multi_branch_fix = multi_branch_fix
        self.Frame_shifts = Frame_shifts

        self.blocks=MaxPool1dLin_scales(num_scale, feature_F_dim, emb_dim, flag_pool).blocks

        base_model = MaxPool1d_blstmlinear_scales(num_scale, feature_F_dim, emb_dim, blstm_layers, flag_pool)
        self.post_nets_seg = base_model.post_nets_seg
        for i in range(num_scale):
            if(str(self.Frame_shifts[i]) in self.multi_branch_fix):
                freeze_by_names_deep(self, f"disc_{i}")

        self.post_nets_utt = base_model.post_nets_utt
        if('utt' in self.multi_branch_fix):
            freeze_by_names(self, "post_nets_utt")


####Gmlp
class gMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, seq_len: int, gmlp_layers = 1, batch_first=True,
            flag_pool = 'None'):
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

        if(d_ffn > 0):
            pass
        elif(d_ffn < 0):
            #if emb_dim <0, we will reduce dim by emb_dim. like -2 will be dim/2
            d_ffn = int(d_model / abs(d_ffn))

        layers = []
        for i in range(gmlp_layers):
            layers.append(GMLPBlock(d_model, d_ffn, seq_len))
        self.layers = nn.Sequential(*layers)

        if(self.flag_pool=='sap'):
            self.pool=nii_nn.SelfWeightedPooling(self.dim, mean_only=True)


        self.fc = nn.Linear(d_model, d_ffn, bias=False)

    def forward(self, x):

        if(self.batch_first):
            x = x.permute(1, 0, 2)
            x = self.layers(x)
            x = x.permute(1, 0, 2)
        else:
            x = self.layers(x)

        #pool for utt
        if(self.flag_pool == "ap"): #average pooling
            x = x.mean(1)
            #nn.AdaptiveAvgPool2d(1)(x)
        elif(self.flag_pool == "sap"):
            x = self.pool(x)
        else:
            pass

        x = self.fc(x)

        return x    

class SSL_BACKEND_MaxPool1d_gmlp_scales(MaxPool1d_scales):
    def __init__(self, num_scale=5, feature_F_dim=60, emb_dim = 256, seq_len = 2001, gmlp_layers=1, batch_first=True, flag_pool = 'ap',
            multi_branch_fix = '', Frame_shifts = [2,4,8,16,32,64] ):

        super(MaxPool1d_scales, self).__init__()

        self.multi_branch_fix = multi_branch_fix
        self.Frame_shifts = Frame_shifts

        self.blocks=MaxPool1d_scales(num_scale, feature_F_dim, emb_dim, flag_pool).blocks

        self.post_nets_seg = nn.ModuleDict()
        for i in range(num_scale):
            dim = feature_F_dim // pow(2,i) 
            self.post_nets_seg[f"disc_{i}"] = gMLP(dim, emb_dim, seq_len // pow(2,i) , gmlp_layers = gmlp_layers, 
                                                   batch_first=batch_first, flag_pool='None')
            if(str(self.Frame_shifts[i]) in self.multi_branch_fix):
                freeze_by_names_deep(self, f"disc_{i}")
        #utt    
        self.post_nets_utt = gMLP(dim, emb_dim, seq_len // pow(2,i) , gmlp_layers = gmlp_layers, 
                                                   batch_first=batch_first, flag_pool=flag_pool)
        if('utt' in self.multi_branch_fix):
            freeze_by_names(self, "post_nets_utt")

class SSL_BACKEND_MaxPool1dLin_gmlp_scales(MaxPool1d_scales):
    def __init__(self, num_scale=5, feature_F_dim=60, emb_dim = 256, seq_len = 2001, gmlp_layers=1, batch_first=True, flag_pool = 'ap',
            multi_branch_fix = '', Frame_shifts = [2,4,8,16,32,64] ):

        super(MaxPool1d_scales, self).__init__()
        self.multi_branch_fix = multi_branch_fix
        self.Frame_shifts = Frame_shifts

        self.blocks=MaxPool1dLin_scales(num_scale, feature_F_dim, emb_dim, flag_pool,
                Frame_shifts = Frame_shifts).blocks

        self.post_nets_seg = nn.ModuleDict()
        for i in range(num_scale):
            dim = feature_F_dim // pow(2,i) 
            self.post_nets_seg[f"disc_{i}"] = gMLP(dim, emb_dim, seq_len // pow(2,i) , gmlp_layers = gmlp_layers, 
                                                   batch_first=batch_first, flag_pool='None')
            if(str(self.Frame_shifts[i]) in self.multi_branch_fix):
                freeze_by_names_deep(self, f"disc_{i}")
        #utt    
        self.post_nets_utt = gMLP(dim, emb_dim, seq_len // pow(2,i) , gmlp_layers = gmlp_layers, 
                                                   batch_first=batch_first, flag_pool=flag_pool)
        if('utt' in self.multi_branch_fix):
            freeze_by_names(self, "post_nets_utt")



def debug():
    t = torch.rand((8, 48, 768))
    #model=MaxPool1dLin_scales(num_scale=7, feature_F_dim=768, flag_pool='ap')
    model = SSL_BACKEND_MaxPool1d_gmlp_scales(num_scale=7, feature_F_dim=768, emb_dim=-2,
            seq_len=2001, gmlp_layers = 1, batch_first=True, flag_pool='ap' )
    model_ml = SSL_BACKEND_MaxPool1dLin_gmlp_scales(num_scale=7, feature_F_dim=768, emb_dim=-2,
            seq_len=2001, gmlp_layers = 1, batch_first=True, flag_pool='ap' )
    o = model(t)

    print(o.shape)


if __name__ == '__main__':
    debug()



class SelfWeightedPooling(torch_nn.Module):
    """ SelfWeightedPooling module
    Inspired by
    https://github.com/joaomonteirof/e2e_antispoofing/blob/master/model.py
    To avoid confusion, I will call it self weighted pooling
    
    Using self-attention format, this is similar to softmax(Query, Key)Value
    where Query is a shared learnarble mm_weight, Key and Value are the input
    Sequence.

    l_selfpool = SelfWeightedPooling(5, 1, False)
    with torch.no_grad():
        input_data = torch.rand([3, 10, 5])
        output_data = l_selfpool(input_data)
    """
    def __init__(self, feature_dim, num_head=1, mean_only=False):
        """ SelfWeightedPooling(feature_dim, num_head=1, mean_only=False)
        Attention-based pooling
        
        input (batchsize, length, feature_dim) ->
        output 
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
        
        args
        ----
          feature_dim: dimension of input tensor
          num_head: number of heads of attention
          mean_only: whether compute mean or mean with std
                     False: output will be (batchsize, feature_dim*2)
                     True: output will be (batchsize, feature_dim)
        """
        super(SelfWeightedPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mean_only = mean_only
        self.noise_std = 1e-5
        self.num_head = num_head

        # transformation matrix (num_head, feature_dim)
        self.mm_weights = torch_nn.Parameter(
            torch.Tensor(num_head, feature_dim), requires_grad=True)
        torch_init.kaiming_uniform_(self.mm_weights)
        return
    
    def _forward(self, inputs):
        """ output, attention = forward(inputs)
        inputs
        ------
          inputs: tensor, shape (batchsize, length, feature_dim)
        
        output
        ------
          output: tensor
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
          attention: tensor, shape (batchsize, length, num_head)
        """        
        # batch size
        batch_size = inputs.size(0)
        # feature dimension
        feat_dim = inputs.size(2)
        
        # input is (batch, legth, feature_dim)
        # change mm_weights to (batchsize, feature_dim, num_head)
        # weights will be in shape (batchsize, length, num_head)
        weights = torch.bmm(inputs, 
                            self.mm_weights.permute(1, 0).contiguous()\
                            .unsqueeze(0).repeat(batch_size, 1, 1))
        
        # attention (batchsize, length, num_head)
        attentions = torch_nn_func.softmax(torch.tanh(weights),dim=1)        
        
        # apply attention weight to input vectors
        if self.num_head == 1:
            # We can use the mode below to compute self.num_head too
            # But there is numerical difference.
            #  original implementation in github
            
            # elmentwise multiplication
            # weighted input vector: (batchsize, length, feature_dim)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            # weights_mat = (batch * length, feat_dim, num_head)
            weighted = torch.bmm(
                inputs.view(-1, feat_dim, 1), 
                attentions.view(-1, 1, self.num_head))
            
            # weights_mat = (batch, length, feat_dim * num_head)
            weighted = weighted.view(batch_size, -1, feat_dim * self.num_head)
            
        # pooling
        if self.mean_only:
            # only output the mean vector
            representations = weighted.sum(1)
        else:
            # output the mean and std vector
            noise = self.noise_std * torch.randn(
                weighted.size(), dtype=weighted.dtype, device=weighted.device)

            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)
            # concatenate mean and std
            representations = torch.cat((avg_repr,std_repr),1)
        # done
        return representations, attentions
    
    def forward(self, inputs):
        """ output = forward(inputs)
        inputs
        ------
          inputs: tensor, shape (batchsize, length, feature_dim)
        
        output
        ------
          output: tensor
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
        """
        output, _ = self._forward(inputs)
        return output

    def debug(self, inputs):
        return self._forward(inputs)



