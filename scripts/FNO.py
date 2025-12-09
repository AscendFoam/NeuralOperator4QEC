"""
FNO.py - 傅里叶神经算子 (Fourier Neural Operator) 用于GKP态去噪

实现特性：
1. 标准2D傅里叶卷积层
2. 残差连接增强训练稳定性
3. 可配置的网络深度和宽度
4. 支持多种激活函数
5. 可选的坐标嵌入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SpectralConv2d(nn.Module):
    """
    2D傅里叶卷积层

    在傅里叶空间中进行线性变换，保留低频模式
    这使得FNO能够高效地学习全局模式
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            modes1: 第一维度保留的傅里叶模式数
            modes2: 第二维度保留的傅里叶模式数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # 复数权重初始化
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """复数矩阵乘法"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, channels, height, width) 实数张量

        Returns:
            (batch, out_channels, height, width) 实数张量
        """
        batchsize = x.shape[0]
        height, width = x.shape[-2], x.shape[-1]

        # 2D实FFT
        x_ft = torch.fft.rfft2(x)

        # 输出傅里叶空间张量
        out_ft = torch.zeros(
            batchsize, self.out_channels, height, width // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # 确保不超出实际频率范围
        modes1 = min(self.modes1, x_ft.shape[-2] // 2)
        modes2 = min(self.modes2, x_ft.shape[-1])

        # 处理正频率部分 (左上角)
        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2],
            self.weights1[:, :, :modes1, :modes2]
        )

        # 处理负频率部分 (左下角，由于FFT的对称性)
        out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(
            x_ft[:, :, -modes1:, :modes2],
            self.weights2[:, :, :modes1, :modes2]
        )

        # 逆FFT回到空间域
        x = torch.fft.irfft2(out_ft, s=(height, width))

        return x


class FNOBlock(nn.Module):
    """
    FNO基本块：傅里叶卷积 + 局部卷积 + 残差连接

    结构: x -> [SpectralConv + Conv1x1] -> Activation -> out
                    ↓_________________________↑ (残差)
    """

    def __init__(self, channels: int, modes1: int, modes2: int,
                 activation: str = 'gelu', use_residual: bool = True):
        super().__init__()

        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.local_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.use_residual = use_residual

        # 激活函数选择
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'silu':
            self.activation = F.silu
        else:
            self.activation = F.gelu

        # 可选的LayerNorm
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 傅里叶路径 + 局部路径
        x = self.spectral_conv(x) + self.local_conv(x)
        x = self.norm(x)
        x = self.activation(x)

        # 残差连接
        if self.use_residual:
            x = x + identity

        return x


class FNO2d_Denoiser(nn.Module):
    """
    用于GKP态去噪的2D傅里叶神经算子

    网络结构：
    1. 输入投影层 (1 -> width)
    2. N个FNO块
    3. 输出投影层 (width -> 1)

    特点：
    - 保持空间分辨率（无池化）
    - 全局感受野（通过傅里叶变换）
    - 支持坐标嵌入
    """

    def __init__(self,
                 modes: int = 12,
                 width: int = 32,
                 n_layers: int = 4,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 activation: str = 'gelu',
                 use_residual: bool = True,
                 use_coord_embed: bool = False,
                 output_activation: str = 'sigmoid'):
        """
        Args:
            modes: 保留的傅里叶模式数
            width: 隐藏层通道数
            n_layers: FNO块数量
            in_channels: 输入通道数
            out_channels: 输出通道数
            activation: 激活函数类型
            use_residual: 是否使用残差连接
            use_coord_embed: 是否添加坐标嵌入
            output_activation: 输出激活函数
        """
        super().__init__()

        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.use_coord_embed = use_coord_embed

        # 坐标嵌入增加2个通道 (x, p)
        input_dim = in_channels + 2 if use_coord_embed else in_channels

        # 输入投影
        self.input_proj = nn.Conv2d(input_dim, width, kernel_size=1)

        # FNO块
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes, modes, activation, use_residual)
            for _ in range(n_layers)
        ])

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1)
        )

        # 输出激活
        if output_activation == 'sigmoid':
            self.output_activation = torch.sigmoid
        elif output_activation == 'tanh':
            self.output_activation = torch.tanh
        else:
            self.output_activation = lambda x: x

        # 坐标网格缓存
        self._coord_cache = {}

    def _get_coord_grid(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """生成坐标网格"""
        key = (shape, device)
        if key not in self._coord_cache:
            batch, _, h, w = shape
            # 创建归一化坐标 [-1, 1]
            x = torch.linspace(-1, 1, w, device=device)
            p = torch.linspace(-1, 1, h, device=device)
            xx, pp = torch.meshgrid(p, x, indexing='ij')
            coords = torch.stack([xx, pp], dim=0)  # (2, H, W)
            coords = coords.unsqueeze(0).expand(batch, -1, -1, -1)  # (B, 2, H, W)
            self._coord_cache[key] = coords
        return self._coord_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, 1, height, width) 噪声Wigner函数

        Returns:
            (batch, 1, height, width) 去噪后的Wigner函数
        """
        # 坐标嵌入
        if self.use_coord_embed:
            coords = self._get_coord_grid(x.shape, x.device)
            x = torch.cat([x, coords], dim=1)

        # 投影到高维空间
        x = self.input_proj(x)

        # FNO块处理
        for block in self.fno_blocks:
            x = block(x)

        # 投影回输出空间
        x = self.output_proj(x)

        # 输出激活
        x = self.output_activation(x)

        return x

    def count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FNO2d_Classifier(nn.Module):
    """
    用于GKP逻辑态分类的FNO

    在去噪的同时进行逻辑态识别
    """

    def __init__(self,
                 modes: int = 12,
                 width: int = 32,
                 n_layers: int = 4,
                 n_classes: int = 4):
        super().__init__()

        # 共享的特征提取器
        self.input_proj = nn.Conv2d(1, width, kernel_size=1)

        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes, modes, 'gelu', True)
            for _ in range(n_layers)
        ])

        # 去噪头
        self.denoise_head = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 分类头
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width, n_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            denoised: (B, 1, H, W) 去噪结果
            logits: (B, n_classes) 分类logits
        """
        x = self.input_proj(x)

        for block in self.fno_blocks:
            x = block(x)

        denoised = self.denoise_head(x)
        logits = self.classifier_head(x)

        return denoised, logits


def create_fno_denoiser(config: dict = None) -> FNO2d_Denoiser:
    """
    工厂函数：创建FNO去噪器

    Args:
        config: 配置字典，可包含 modes, width, n_layers 等参数

    Returns:
        FNO2d_Denoiser 实例
    """
    default_config = {
        'modes': 12,
        'width': 32,
        'n_layers': 4,
        'activation': 'gelu',
        'use_residual': True,
        'use_coord_embed': False,
        'output_activation': 'sigmoid'
    }

    if config:
        default_config.update(config)

    return FNO2d_Denoiser(**default_config)


if __name__ == '__main__':
    # 测试FNO模型
    print("Testing FNO2d_Denoiser...")

    model = FNO2d_Denoiser(modes=12, width=32, n_layers=4)
    print(f"Model parameters: {model.count_parameters():,}")

    # 测试前向传播
    x = torch.randn(4, 1, 64, 64)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")

    # 测试分类器
    print("\nTesting FNO2d_Classifier...")
    classifier = FNO2d_Classifier(modes=12, width=32, n_layers=4, n_classes=4)
    denoised, logits = classifier(x)
    print(f"Denoised shape: {denoised.shape}")
    print(f"Logits shape: {logits.shape}")
