import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SEModule(nn.Module):
    def __init__(self, channels, SE_ratio=8):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // SE_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // SE_ratio, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class ASTP(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super(ASTP, self).__init__()
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat([mean, std], dim=1)

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, scale=8, SE_ratio=8):
        super(Bottle2neck, self).__init__()
        width = planes // scale
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs, bns, wsum = [], [], []
        pad = (kernel_size // 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=(kernel_size, 1), dilation=(dilation, 1), padding=(pad, 0)))
            bns.append(nn.BatchNorm2d(width))
            wsum.append(nn.Parameter(torch.ones(1, 1, 1, i + 2) * (1 / (i + 2)), requires_grad=True))
        self.weighted_sum = nn.ParameterList(wsum)
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes, SE_ratio)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out).unsqueeze(-1)  # bz c T 1

        spx = torch.split(out, self.width, 1)
        sp = spx[self.nums]
        for i in range(self.nums):
            sp = torch.cat((sp, spx[i]), -1)

            sp = self.bns[i](self.relu(self.convs[i](sp)))
            sp_s = sp * self.weighted_sum[i]
            sp_s = torch.sum(sp_s, dim=-1, keepdim=False)

            if i == 0:
                out = sp_s
            else:
                out = torch.cat((out, sp_s), 1)
        out = torch.cat((out, spx[self.nums].squeeze(-1)), 1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out

class SSL_BACKEND_utt_only_nes2net(nn.Module):
    def __init__(self, feat_dim=1024, Nes_ratio=[8, 8], dilation=1, embed_dim=2048, SE_ratio=1, pool_func='ASTP'):
        super().__init__()
        self.Nes_ratio = Nes_ratio[0]
        assert feat_dim % Nes_ratio[0] == 0
        C = feat_dim // Nes_ratio[0]
        Build_in_Res2Nets, bns = [], []
        for _ in range(Nes_ratio[0] - 1):
            Build_in_Res2Nets.append(Bottle2neck(C, C, kernel_size=3, dilation=dilation, scale=Nes_ratio[1], SE_ratio=SE_ratio))
            bns.append(nn.BatchNorm1d(C))
        self.Build_in_Res2Nets = nn.ModuleList(Build_in_Res2Nets)
        self.bns = nn.ModuleList(bns)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.relu = nn.ReLU()
        self.pooling = ASTP(in_dim=feat_dim, bottleneck_dim=128, global_context_att=False)
        # self.fc = nn.Linear(feat_dim * 2, embed_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        spx = torch.split(x, x.size(1) // self.Nes_ratio, 1)
        for i in range(self.Nes_ratio - 1):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](self.relu(self.Build_in_Res2Nets[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        out = torch.cat((out, spx[-1]), 1)
        out = self.relu(self.bn(out))
        out = self.pooling(out)
        # out = self.fc(out)
        return out

def debug_backend():
    # input：batch_size = 8, seq_len = 200, feat_dim = 1024
    batch_size = 8
    seq_len = 200
    feat_dim = 1024
    embed_dim = 128

    x = torch.randn(batch_size, seq_len, feat_dim)

    model = SSL_BACKEND_utt_only_nes2net(feat_dim=feat_dim)

    with torch.no_grad():
        out = model(x)

    print("Input shape:", x.shape)       # B x T x D
    print("Output shape:", out.shape)    # B x embed_dim，如 (8, 128)

if __name__ == "__main__":
    debug_backend()
