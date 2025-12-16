import torch.nn as nn
import torch.nn.functional as F
import torch

class SpectralConv2d(nn.Module):
    """
    2D Fourier Layer。
    这是 FNO 的核心，在频域进行卷积操作。
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Fourier modes to keep along axis 1
        self.modes2 = modes2 # Fourier modes to keep along axis 2

        self.scale = (1 / (in_channels * out_channels))
        # 权重矩阵：复数张量
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # 1. 傅里叶变换到频域
        x_ft = torch.fft.rfft2(x)

        # 2. 频域滤波 (只保留低频 modes，但在 GKP 中我们需要适当增加 modes 以保留格点特征)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # 处理左上角低频部分
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # 处理左下角（因为实数FFT的共轭对称性）
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 3. 傅里叶逆变换回空域
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2dDecoder(nn.Module):
    """
    GKP Syndrome Decoder based on FNO.
    Input: Wigner Function (B, 1, 32, 32)
    Output: Displacement Vector (B, 2) -> (u, v)
    """
    def __init__(self, modes=12, width=32):
        super(FNO2dDecoder, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width

        # Lifting Layer: 将输入映射到高维特征空间
        self.p = nn.Linear(1, self.width) 
        
        # Fourier Layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # Residual connections (1x1 convs)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        # Pooling & Projection to scalar output
        # 我们需要从整张图的特征中提取出全局的位移量
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2) # Output: (u, v) correction

    def forward(self, x):
        # x shape: (Batch, 1, Grid, Grid)
        batch_size = x.shape[0]
        grid_size = x.shape[2]
        
        # 调整维度以适应 Linear 层: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.p(x) # Lifting
        x = x.permute(0, 3, 1, 2) # Back to (B, C, H, W)

        # Layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        # Layer 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # Global Pooling: 聚合所有空间信息
        # FNO 对于捕捉全局位移（Global shift）非常有效，因为位移在频域表现为相移
        x = self.pool(x) # (B, Width, 1, 1)
        x = x.view(batch_size, -1)
        
        # Decoding Head
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x