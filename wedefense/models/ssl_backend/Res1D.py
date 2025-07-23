import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))

class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.width)
            for _ in range(self.nums)
        ])

    def forward(self, x):
        spx = torch.split(x, self.width, dim=1)
        out = []
        sp = spx[0]
        for i in range(self.nums):
            if i >= 1:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)

class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)

class SE_Res2Block(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()
        self.block = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1),
            Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale),
            Conv1dReluBn(channels, channels, kernel_size=1),
            SE_Connect(channels)
        )

    def forward(self, x):
        return x + self.block(x)
class Res1D(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.res1D = nn.Sequential(
            Conv1dReluBn(channels,
                         channels,
                         kernel_size=1,
                         stride=1,
                         padding=0),
            Conv1dReluBn(channels,
                         channels,
                         kernel_size=3,
                         stride=1,
                         padding=1),
            Conv1dReluBn(channels,
                         channels,
                         kernel_size=1,
                         stride=1,
                         padding=0), SE_Connect(channels))

    def forward(self, x):
        return x + self.res1D(x)
class SSL_BACKEND_utt_only_res1d(nn.Module):
    def __init__(self, feat_dim=1024, embed_dim=128):
        super().__init__()
        input_dim = feat_dim 
        self.res1D = Res1D(feat_dim)
        self.fc = nn.Conv1d(input_dim,
                  embed_dim,
                  kernel_size=1,
                  stride=1,
                  padding=0)

    def forward(self, x):
        """
        Args:
            x: Tensor (B x T x D), output from SSL frontend
        Returns:
            Tensor (B x embed_dim)
        """
        x = x.transpose(1, 2)          # B x D x T
        x = self.res1D(x)          # B x D x T
        x = self.fc(x)              # B x 128 x T
        x = torch.mean(x, dim=2, keepdims=True)
        x = x.squeeze(-1)
        return x

def debug_backend():
     # B x T x D
    batch_size = 8
    seq_len = 2001
    input_dim = 1024  # 

    x = torch.rand(batch_size, seq_len, input_dim)

    model = SSL_BACKEND_utt_only_res1d(feat_dim=input_dim, embed_dim=128)
    out = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  #  (B x embed_dim) -> (8, 128)

if __name__ == '__main__':
    debug_backend()
