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

class ASTP(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super().__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x, mean_only=True):
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            x_in = torch.cat((x, mean, std), dim=1)
        else:
            x_in = x

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-7))
        return mean if mean_only else torch.cat([mean, std], dim=1)

class SSL_BACKEND_utt_only_res1d(nn.Module):
    def __init__(self, feat_dim=1024, embed_dim=128, bottleneck_dim=128):
        super().__init__()
        input_dim = feat_dim
        self.res_block = SE_Res2Block(
            channels=input_dim,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8
        )
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=1)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(128, embed_dim, kernel_size=1)
        self.pool = ASTP(in_dim=embed_dim, bottleneck_dim=bottleneck_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor (B x T x D), output from SSL frontend
        Returns:
            Tensor (B x embed_dim)
        """
        x = x.transpose(1, 2)          # B x D x T
        x = self.res_block(x)          # B x D x T
        x = self.conv1(x)              # B x 128 x T
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)              # B x embed_dim x T
        x = self.pool(x)               # B x embed_dim
        return x

def debug_backend():
    # 假设输入来自 SSL（如 wav2vec2）输出，维度为 B x T x D
    batch_size = 8
    seq_len = 2001
    input_dim = 1024  # SSL模型如wav2vec2-large的输出维度

    x = torch.rand(batch_size, seq_len, input_dim)

    model = SSL_BACKEND_utt_only_res1d(feat_dim=input_dim, embed_dim=128, bottleneck_dim=128)
    out = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  # 应为 (B x embed_dim)，如 (8, 128)

if __name__ == '__main__':
    debug_backend()
