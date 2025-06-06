# Copyright (c)  2024 Jiafeng Zhong
#                2025 You Zhang (you.zhang@rochester.edu)
#              
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
BAM as the backend for SSL models for spoof localization tasks.

From the paper: Enhancing Partially Spoofed Audio Localization with Boundary-aware Attention Mechanism
Author: Jiafeng Zhong, Bin Li, Jiangyan Yi
Link: https://www.isca-archive.org/interspeech_2024/zhong24_interspeech.pdf
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import random


class SSL_BACKEND_BAM(nn.Module):
    def __init__(self, resolution=0.16, feat_dim=1024, pool_head_num=1, gap_layer_num=2, gap_head_num=1, local_channel_dim=32, embed_dim=2048) -> None:
        super(SSL_BACKEND_BAM, self).__init__()
        self.name = "BAM"
        self.feat_dim = feat_dim
        
        self.pool_head_num = pool_head_num
        self.att_pool = SelfWeightedPooling(feat_dim, num_head=pool_head_num, mean_only=True)
        self.selu = nn.SELU()

        self.inter_layer = inter_frame_attention(in_dim=feat_dim, out_dim=feat_dim, head_num=gap_head_num)
        self.intra_layer = intra_frame_attention(in_channel=local_channel_dim, in_dim=feat_dim, out_dim=feat_dim)

        self.b_output_layer = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=1),
            nn.Sigmoid()
        )

        self.pool_frame_num = int(resolution // 0.02)
        
        self.gap_layers= nn.ModuleList([
            MessageControlGraphAttentionLayer(in_dim=feat_dim, out_dim=feat_dim, head_num=gap_head_num)
            for _ in range(gap_layer_num)])
        
        self.boundary_proj = nn.Linear(in_features=embed_dim, out_features=feat_dim)
        self.out_layer = nn.Linear(in_features=embed_dim,out_features=2)


    def forward(self, x, ret_emb=False):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [Batch, Dim, Frame_len, Nb_Layer]

        Returns:
            outs: Output tensor after passing through the network
        """

        # x = F.pad(x,(0,256),mode='constant', value=0)
        # x = self.ssl_layer(x)["hidden_states"][-1]

        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer] 
        
        b, f, d = x.size() # (B, 150, 1024)
        # x = x.view(-1, self.pool_frame_num, d)
        # x = self.att_pool(x)
        # x = x.view(b,-1, d * self.pool_head_num)
        # x = self.proj(x)

        # boundary
        f_inter = self.inter_layer(x)
        f_intra = self.intra_layer(x)
        b_embedding = torch.cat([f_inter, f_intra], dim=-1)
        b_pred = self.b_output_layer(b_embedding).squeeze(-1)
        binary = torch.where(b_pred.detach() > 0.5, 1, 0)

        # spoof
        s_embedding = x
        for layer in self.gap_layers:
            s_embedding = layer(s_embedding, binary)

        # fusion              
        b_embedding_detach = self.selu(self.boundary_proj(b_embedding.detach()))                                                                                                                                                            
        embedding = torch.cat([s_embedding,b_embedding_detach], dim=-1)
        output = self.out_layer(embedding)

        if ret_emb:
            return embedding
        else:
            return output, b_pred


class inter_frame_attention(nn.Module):
    """ attention among ssl encoder output features."""
    def __init__(self,in_dim, out_dim, head_num):
        super(inter_frame_attention, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, head_num)

        # project
        self.proj_with_att = nn.Linear(head_num * in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # activate
        self.act = nn.SELU(inplace=True)

    def forward(self,x):
        att_map = self._derive_att_map(x)
        att_map = F.softmax(att_map, dim=-2)
        x = self._project(x, att_map)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _project(self, x, att_map):
        x1 = torch.matmul(att_map.permute(0,3,1,2).contiguous(), x.unsqueeze(1).expand(-1,self.head_num,-1,-1))
        x1 = self.proj_with_att(x1.permute(0,2,3,1).flatten(start_dim=2))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        return att_map

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror


class intra_frame_attention(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_channel, in_dim, out_dim):
        super(intra_frame_attention, self).__init__()
        self.resnet = ResNet1D(BottleNeck,[2,2,2,2], in_channel=in_channel)
        self.proj_layer = nn.Linear(in_features=1024*int(in_dim/32), out_features=out_dim)

    def forward(self,x):
        m_batch, T, D = x.size()
        out = self.resnet(x.view(-1,1,D))
        out = self.proj_layer(out.view(m_batch,T,-1))
        return out


class SelfWeightedPooling(nn.Module):

    """ SelfWeightedPooling module
    Inspired by
    https://github.com/joaomonteirof/e2e_antispoofing/blob/master/model.py
    To avoid confusion, I will call it self weighted pooling
    
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
        self.mm_weights = nn.Parameter(torch.Tensor(feature_dim, num_head), requires_grad=True)
        nn.init.kaiming_uniform_(self.mm_weights)
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
        weights = torch.bmm(inputs, self.mm_weights.unsqueeze(0).repeat(batch_size, 1, 1))
        
        # attention (batchsize, length, num_head)
        attentions = nn.functional.softmax(torch.tanh(weights),dim=1)        
        
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
            #    inputs.view(-1, feat_dim, 1), zl, error
            #    RuntimeError: view size is not compatible with input tensor's size and stride 
            #    (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            weighted = torch.bmm(
                inputs.reshape(-1, feat_dim, 1), 
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


class BottleNeck(nn.Module):
    expansion = 2
    def __init__(self,in_channel,channel,stride=1,downsample=None):
        super().__init__()

        self.conv1=nn.Conv1d(in_channel,channel,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm1d(channel)

        self.conv2=nn.Conv1d(channel,channel,kernel_size=3,padding=1,bias=False,stride=1)
        self.bn2=nn.BatchNorm1d(channel)

        self.conv3=nn.Conv1d(channel,channel*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm1d(channel*self.expansion)

        self.relu=nn.ReLU(False)

        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x

        out=self.relu(self.bn1(self.conv1(x))) #bs,c,h,w
        out=self.relu(self.bn2(self.conv2(out))) #bs,c,h,w
        out=self.relu(self.bn3(self.conv3(out))) #bs,4c,h,w

        if(self.downsample != None):
            residual=self.downsample(residual)

        out = out + residual
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(self,block,layers,in_channel):
        super().__init__()
        #定义输入模块的维度
        self.in_channel=in_channel
        ### stem layer
        self.conv1=nn.Conv1d(1,in_channel,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm1d(in_channel)
        self.relu=nn.ReLU(False)
        self.maxpool=nn.MaxPool1d(kernel_size=3,stride=2,padding=0,ceil_mode=True)

        ### main layer
        self.layer1=self._make_layer(block,64,layers[0])
        self.layer2=self._make_layer(block,128,layers[1],stride=2)
        self.layer3=self._make_layer(block,256,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)

    def forward(self,x):
        ##stem layer
        out=self.relu(self.bn1(self.conv1(x))) #bs,112,112,64
        out=self.maxpool(out) #bs,56,56,64

        ##layers:
        out=self.layer1(out) #bs,56,56,64*4
        out=self.layer2(out) #bs,28,28,128*4
        out=self.layer3(out) #bs,14,14,256*4
        out=self.layer4(out) #bs,7,7,512*4

        return out

    def _make_layer(self,block,channel,blocks,stride=1):
        # downsample 主要用来处理H(x)=F(x)+x中F(x)和x的channel维度不匹配问题，即对残差结构的输入进行升维，在做残差相加的时候，必须保证残差的纬度与真正的输出维度（宽、高、以及深度）相同
        # 比如步长！=1 或者 in_channel!=channel&self.expansion
        downsample = None
        if(stride!=1 or self.in_channel!=channel*block.expansion):
            self.downsample=nn.Conv1d(self.in_channel,channel*block.expansion,stride=stride,kernel_size=1,bias=False)
        #第一个conv部分，可能需要downsample
        layers=[]
        layers.append(block(self.in_channel,channel,downsample=self.downsample,stride=stride))
        self.in_channel=channel*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_channel,channel))
        return nn.Sequential(*layers)


class MessageControlGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, head_num, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, head_num)

        # project
        self.proj_with_att = nn.Linear(head_num * in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x, boundary):
        '''
        x   :(#bs, #node, #dim)
        '''
        # derive attention map
        att_map = self._derive_att_map(x)

        # message control
        att_map = self._message_control_matrix(boundary, att_map)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _message_control_matrix(self, batch_boundary, att_map):
        if batch_boundary is not None:
            b, t = batch_boundary.size()
            matrix_all = torch.zeros((b, t, t, self.head_num), dtype=batch_boundary.dtype, device=batch_boundary.device)

            for i in range(b):
                boundary = batch_boundary[i]
                for j in range(t):
                    mask = (1 - boundary[j:]).cumprod(dim=0).unsqueeze(1)
                    matrix_all[i, j, j:, :] = mask.repeat(1,self.head_num)
                    matrix_all[i, j:, j, :] = mask.repeat(1,self.head_num)

            matrix_all[:, range(t), range(t), :] = 1
            att_map = att_map * matrix_all.to(att_map.device)

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, head_num)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        return att_map

    def _project(self, x, att_map):
        x1 = torch.matmul(att_map.permute(0,3,1,2).contiguous(), x.unsqueeze(1).expand(-1,self.head_num,-1,-1))
        x1 = self.proj_with_att(x1.permute(0,2,3,1).flatten(start_dim=2))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


if __name__ == "__main__":
    # Example usage
    model = BAM(resolution=0.16, embed_dim=1024, pool_head_num=1, gap_layer_num=2, gap_head_num=1, local_channel_dim=32)
    x = torch.randn(8, 256, 1024)  # Example input tensor (batch_size, frame_length, feature_dim)
    output, b_pred = model(x)
    print("Output shape:", output.shape)  # Should be (8, 2) for binary classification
    print("Boundary prediction shape:", b_pred.shape)  # Should be (8, frame_length)
