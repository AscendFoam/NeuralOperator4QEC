"""
UNet.py - U-Net模型用于GKP态去噪

实现特性：
1. 标准U-Net编码器-解码器结构
2. Skip连接保留细节
3. 周期性边界条件（适合相空间）
4. 可配置的深度和通道数
5. 支持注意力机制（可选）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class DoubleConv(nn.Module):
    """
    双卷积块: (Conv -> BN -> ReLU) x 2

    支持周期性padding以适应相空间的周期性边界条件
    """

    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: int = None,
                 use_circular_padding: bool = True):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        padding_mode = 'circular' if use_circular_padding else 'zeros'

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,
                     padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,
                     padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """下采样块: MaxPool -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int,
                 use_circular_padding: bool = True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_circular_padding=use_circular_padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样块: Upsample -> Concat -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int,
                 bilinear: bool = True, use_circular_padding: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels,
                                  mid_channels=in_channels // 2,
                                  use_circular_padding=use_circular_padding)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,
                                  use_circular_padding=use_circular_padding)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: 来自下层的特征
            x2: 来自编码器的skip连接
        """
        x1 = self.up(x1)

        # 处理尺寸不匹配（如果有的话）
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])

        # 连接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    注意力门控块

    用于增强skip连接中的相关特征
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: 门控信号通道数
            F_l: 输入特征通道数
            F_int: 中间通道数
        """
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: 门控信号（来自解码器）
            x: 输入特征（来自编码器skip连接）
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 如果尺寸不匹配，调整g1
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UNet_Denoiser(nn.Module):
    """
    用于GKP态去噪的U-Net模型

    网络结构：
    - 编码器：逐步下采样，提取多尺度特征
    - 瓶颈层：最深层特征处理
    - 解码器：逐步上采样，融合skip连接
    - 输出层：投影回单通道

    特点：
    - 周期性边界条件
    - 可选的注意力机制
    - 灵活的深度配置
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 32,
                 depth: int = 4,
                 bilinear: bool = True,
                 use_circular_padding: bool = True,
                 use_attention: bool = False,
                 output_activation: str = 'sigmoid'):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            base_channels: 基础通道数（后续层会翻倍）
            depth: 网络深度（下采样次数）
            bilinear: 是否使用双线性上采样
            use_circular_padding: 是否使用周期性padding
            use_attention: 是否使用注意力机制
            output_activation: 输出激活函数
        """
        super().__init__()

        self.depth = depth
        self.use_attention = use_attention

        # 计算各层通道数
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        # 限制最大通道数
        channels = [min(c, 512) for c in channels]
        self.channels = channels

        # 编码器
        self.inc = DoubleConv(in_channels, channels[0],
                             use_circular_padding=use_circular_padding)

        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            self.down_blocks.append(
                Down(channels[i], channels[i + 1], use_circular_padding)
            )

        # 解码器 - 正确计算通道数
        # 需要跟踪每个up block的输出通道数
        self.up_blocks = nn.ModuleList()
        decoder_in_channels = channels[depth]  # 从最深层开始

        for i in range(depth):
            skip_channels = channels[depth - 1 - i]  # 对应的skip连接通道数
            in_ch = decoder_in_channels + skip_channels  # concat后的通道数
            out_ch = skip_channels  # 输出通道数与skip相同
            self.up_blocks.append(
                Up(in_ch, out_ch, bilinear, use_circular_padding)
            )
            decoder_in_channels = out_ch  # 更新下一层的输入通道数

        # 注意力块（可选）
        if use_attention:
            self.attention_blocks = nn.ModuleList()
            attn_in_channels = channels[depth]
            for i in range(depth):
                skip_channels = channels[depth - 1 - i]
                F_g = attn_in_channels
                F_l = skip_channels
                F_int = max(skip_channels // 2, 1)
                self.attention_blocks.append(
                    AttentionBlock(F_g, F_l, F_int)
                )
                attn_in_channels = skip_channels

        # 输出层
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

        # 输出激活
        if output_activation == 'sigmoid':
            self.output_activation = torch.sigmoid
        elif output_activation == 'tanh':
            self.output_activation = torch.tanh
        else:
            self.output_activation = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, 1, height, width) 噪声输入

        Returns:
            (batch, 1, height, width) 去噪输出
        """
        # 编码器路径 - 保存每一层的特征用于skip连接
        encoder_features = []
        x = self.inc(x)
        encoder_features.append(x)  # channels[0]

        for down in self.down_blocks:
            x = down(x)
            encoder_features.append(x)

        # 此时 encoder_features = [f0, f1, f2, ..., f_depth]
        # x = encoder_features[-1] = f_depth (最深层特征)

        # 解码器路径
        for i, up in enumerate(self.up_blocks):
            # skip连接来自编码器的对应层 (从深到浅)
            skip_idx = self.depth - 1 - i  # depth-1, depth-2, ..., 0
            skip = encoder_features[skip_idx]

            # 应用注意力（如果启用）
            if self.use_attention:
                skip = self.attention_blocks[i](x, skip)

            x = up(x, skip)

        # 输出
        x = self.outc(x)
        x = self.output_activation(x)

        return x

    def count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualUNet_Denoiser(nn.Module):
    """
    残差U-Net变体

    在skip连接的基础上添加全局残差
    适合学习噪声残差而非直接重建
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 32,
                 depth: int = 4,
                 bilinear: bool = True):
        super().__init__()

        self.unet = UNet_Denoiser(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            bilinear=bilinear,
            output_activation='none'  # 残差输出不需要激活
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """学习噪声残差，输出 x - noise"""
        noise = self.unet(x)
        # 残差学习：预测噪声，然后减去
        denoised = x - noise
        # 裁剪到有效范围
        denoised = torch.clamp(denoised, 0, 1)
        return denoised


class LightUNet_Denoiser(nn.Module):
    """
    轻量级U-Net

    减少参数量，适合快速推理
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 16,
                 use_circular_padding: bool = True):
        super().__init__()

        # 简化的编码器
        self.enc1 = DoubleConv(in_channels, base_channels, use_circular_padding=use_circular_padding)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, use_circular_padding=use_circular_padding)
        self.pool2 = nn.MaxPool2d(2)

        # 瓶颈
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4, use_circular_padding=use_circular_padding)

        # 简化的解码器
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(base_channels * 6, base_channels * 2, use_circular_padding=use_circular_padding)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(base_channels * 3, base_channels, use_circular_padding=use_circular_padding)

        # 输出
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        # 解码
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.outc(d1))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unet_denoiser(config: dict = None) -> UNet_Denoiser:
    """
    工厂函数：创建U-Net去噪器

    Args:
        config: 配置字典

    Returns:
        UNet_Denoiser 实例
    """
    default_config = {
        'in_channels': 1,
        'out_channels': 1,
        'base_channels': 32,
        'depth': 4,
        'bilinear': True,
        'use_circular_padding': True,
        'use_attention': False,
        'output_activation': 'sigmoid'
    }

    if config:
        default_config.update(config)

    return UNet_Denoiser(**default_config)


if __name__ == '__main__':
    # 测试U-Net模型
    print("Testing UNet_Denoiser...")

    model = UNet_Denoiser(base_channels=32, depth=4)
    print(f"Model parameters: {model.count_parameters():,}")

    # 测试前向传播
    x = torch.randn(4, 1, 64, 64)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")

    # 测试轻量级版本
    print("\nTesting LightUNet_Denoiser...")
    light_model = LightUNet_Denoiser(base_channels=16)
    print(f"Light model parameters: {light_model.count_parameters():,}")
    y_light = light_model(x)
    print(f"Light output shape: {y_light.shape}")

    # 测试注意力U-Net
    print("\nTesting UNet with Attention...")
    attn_model = UNet_Denoiser(base_channels=32, depth=3, use_attention=True)
    print(f"Attention model parameters: {attn_model.count_parameters():,}")
    y_attn = attn_model(x)
    print(f"Attention output shape: {y_attn.shape}")
