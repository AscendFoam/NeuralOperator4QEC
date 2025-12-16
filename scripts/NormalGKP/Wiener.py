"""
Wiener.py - 传统信号处理方法用于GKP态去噪

实现多种经典去噪方法作为基准对比：
1. 维纳滤波 (Wiener Filter)
2. 频域维纳去卷积 (Wiener Deconvolution)
3. 自适应维纳滤波
4. 高斯滤波 (简单基准)
5. 双边滤波 (保边滤波)
"""

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.fft
import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class WienerConfig:
    """维纳滤波器配置"""
    # 估计的扩散核大小
    kernel_sigma: float = 1.5

    # 信噪比估计
    snr_estimate: float = 10.0

    # 自适应滤波窗口大小
    adaptive_window_size: int = 5

    # 正则化参数
    regularization: float = 0.01


class WienerDenoiser:
    """
    基于维纳滤波的去噪器

    实现多种经典信号处理去噪算法
    """

    def __init__(self, config: WienerConfig = None):
        self.config = config or WienerConfig()

    def _create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """创建高斯卷积核"""
        x = np.arange(-(size // 2), size // 2 + 1)
        kernel_1d = np.exp(-x**2 / (2 * sigma**2))
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d

    def _pad_kernel(self, kernel: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """将核填充到目标大小（用于FFT）"""
        h, w = target_shape
        kh, kw = kernel.shape
        padded = np.zeros((h, w))

        # 将核放在左上角
        padded[:kh, :kw] = kernel

        # 循环移位使核中心在(0,0)
        padded = np.roll(padded, -(kh // 2), axis=0)
        padded = np.roll(padded, -(kw // 2), axis=1)

        return padded

    def wiener_filter(self, image: np.ndarray) -> np.ndarray:
        """
        标准空间域维纳滤波

        使用scipy.signal.wiener实现自适应去噪
        """
        # 归一化输入
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 0:
            image_norm = (image - img_min) / (img_max - img_min + 1e-8)
        else:
            image_norm = image

        # 应用维纳滤波
        denoised = scipy.signal.wiener(
            image_norm,
            mysize=self.config.adaptive_window_size,
            noise=1.0 / self.config.snr_estimate
        )

        # 裁剪到有效范围
        denoised = np.clip(denoised, 0, 1)

        return denoised

    def wiener_deconvolution(self, image: np.ndarray) -> np.ndarray:
        """
        频域维纳去卷积

        假设图像被高斯核模糊，使用维纳逆滤波恢复

        公式: F_hat = F_noisy * H* / (|H|^2 + 1/SNR)
        """
        h, w = image.shape

        # 创建估计的PSF（点扩散函数）
        kernel_size = int(self.config.kernel_sigma * 6) | 1  # 确保是奇数
        kernel_size = min(kernel_size, min(h, w) // 2)
        psf = self._create_gaussian_kernel(kernel_size, self.config.kernel_sigma)

        # 填充PSF到图像大小
        psf_padded = self._pad_kernel(psf, (h, w))

        # FFT
        image_fft = scipy.fft.fft2(image)
        psf_fft = scipy.fft.fft2(psf_padded)

        # 维纳滤波公式
        # W = H* / (|H|^2 + 1/SNR)
        snr = self.config.snr_estimate
        wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft)**2 + 1.0 / snr)

        # 应用滤波
        denoised_fft = image_fft * wiener_filter
        denoised = np.real(scipy.fft.ifft2(denoised_fft))

        # 归一化
        denoised = denoised - denoised.min()
        if denoised.max() > 0:
            denoised = denoised / denoised.max()

        return denoised

    def regularized_deconvolution(self, image: np.ndarray) -> np.ndarray:
        """
        带正则化的去卷积

        使用Tikhonov正则化防止噪声放大
        """
        h, w = image.shape

        # 创建PSF
        kernel_size = int(self.config.kernel_sigma * 6) | 1
        kernel_size = min(kernel_size, min(h, w) // 2)
        psf = self._create_gaussian_kernel(kernel_size, self.config.kernel_sigma)
        psf_padded = self._pad_kernel(psf, (h, w))

        # FFT
        image_fft = scipy.fft.fft2(image)
        psf_fft = scipy.fft.fft2(psf_padded)

        # Tikhonov正则化
        # F_hat = H* F_noisy / (|H|^2 + λ)
        lambda_reg = self.config.regularization
        regularized_filter = np.conj(psf_fft) / (np.abs(psf_fft)**2 + lambda_reg)

        # 应用
        denoised_fft = image_fft * regularized_filter
        denoised = np.real(scipy.fft.ifft2(denoised_fft))

        # 归一化
        denoised = denoised - denoised.min()
        if denoised.max() > 0:
            denoised = denoised / denoised.max()

        return np.clip(denoised, 0, 1)

    def gaussian_filter(self, image: np.ndarray, sigma: float = 0.5) -> np.ndarray:
        """
        简单高斯滤波（低通滤波）

        用于对比，这会进一步模糊图像而非锐化
        """
        denoised = scipy.ndimage.gaussian_filter(image, sigma=sigma)
        return denoised

    def bilateral_filter(self, image: np.ndarray,
                        sigma_spatial: float = 2.0,
                        sigma_range: float = 0.1) -> np.ndarray:
        """
        双边滤波

        保边去噪：在平滑的同时保留边缘
        """
        from scipy.ndimage import uniform_filter

        h, w = image.shape

        # 简化的双边滤波实现
        # 对于每个像素，计算加权平均
        # 权重 = 空间高斯 * 强度高斯

        # 这里使用迭代近似
        output = image.copy()
        window_size = int(sigma_spatial * 3) * 2 + 1

        for _ in range(3):  # 迭代次数
            # 计算局部均值和方差
            local_mean = uniform_filter(output, window_size)
            local_var = uniform_filter(output**2, window_size) - local_mean**2
            local_var = np.maximum(local_var, 1e-8)

            # 自适应权重
            noise_var = np.mean(local_var) * sigma_range
            weight = local_var / (local_var + noise_var)

            # 更新
            output = weight * output + (1 - weight) * local_mean

        return np.clip(output, 0, 1)

    def adaptive_wiener(self, image: np.ndarray) -> np.ndarray:
        """
        自适应维纳滤波

        根据局部统计特性自适应调整滤波强度
        """
        from scipy.ndimage import uniform_filter

        window_size = self.config.adaptive_window_size

        # 计算局部均值和方差
        local_mean = uniform_filter(image, window_size)
        local_sq_mean = uniform_filter(image**2, window_size)
        local_var = local_sq_mean - local_mean**2
        local_var = np.maximum(local_var, 0)

        # 估计噪声方差（取最小的局部方差）
        noise_var = np.mean(local_var) * 0.5

        # 自适应权重
        # 当局部方差大时（有信号），保留更多原始值
        # 当局部方差小时（平坦区域），更多地使用均值
        factor = noise_var / (local_var + noise_var + 1e-8)

        # 滤波
        denoised = local_mean + (1 - factor) * (image - local_mean)

        return np.clip(denoised, 0, 1)

    def denoise(self, image: np.ndarray, method: str = 'wiener') -> np.ndarray:
        """
        去噪主接口

        Args:
            image: 输入图像
            method: 方法选择
                - 'wiener': 空间域维纳滤波
                - 'wiener_deconv': 频域维纳去卷积
                - 'regularized': 正则化去卷积
                - 'adaptive': 自适应维纳
                - 'bilateral': 双边滤波
        """
        if method == 'wiener':
            return self.wiener_filter(image)
        elif method == 'wiener_deconv':
            return self.wiener_deconvolution(image)
        elif method == 'regularized':
            return self.regularized_deconvolution(image)
        elif method == 'adaptive':
            return self.adaptive_wiener(image)
        elif method == 'bilateral':
            return self.bilateral_filter(image)
        else:
            raise ValueError(f"Unknown method: {method}")


def wiener_denoising_batch(noisy_batch: np.ndarray,
                          config: WienerConfig = None,
                          method: str = 'adaptive') -> torch.Tensor:
    """
    批量维纳去噪

    Args:
        noisy_batch: (B, 1, H, W) numpy数组
        config: 维纳滤波配置
        method: 去噪方法

    Returns:
        (B, 1, H, W) PyTorch张量
    """
    denoiser = WienerDenoiser(config)
    batch_size = noisy_batch.shape[0]
    denoised_batch = np.zeros_like(noisy_batch)

    for i in range(batch_size):
        img = noisy_batch[i, 0]
        denoised_batch[i, 0] = denoiser.denoise(img, method)

    return torch.FloatTensor(denoised_batch)


class WienerWrapper:
    """
    维纳滤波器包装类

    提供类似神经网络的接口，便于统一评估
    """

    def __init__(self, config: WienerConfig = None, method: str = 'adaptive'):
        self.config = config or WienerConfig()
        self.method = method
        self.denoiser = WienerDenoiser(self.config)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播接口

        Args:
            x: (B, 1, H, W) PyTorch张量

        Returns:
            (B, 1, H, W) 去噪结果
        """
        # 转换为numpy
        x_np = x.cpu().numpy()

        # 批量去噪
        denoised = wiener_denoising_batch(x_np, self.config, self.method)

        # 移动到相同设备
        return denoised.to(x.device)

    def eval(self):
        """兼容神经网络接口"""
        return self

    def train(self):
        """兼容神经网络接口"""
        return self

    def parameters(self):
        """兼容神经网络接口（返回空列表）"""
        return []

    def count_parameters(self) -> int:
        """无可训练参数"""
        return 0


def create_wiener_denoiser(config: dict = None, method: str = 'adaptive') -> WienerWrapper:
    """
    工厂函数：创建维纳去噪器

    Args:
        config: 配置字典
        method: 去噪方法

    Returns:
        WienerWrapper 实例
    """
    default_config = {
        'kernel_sigma': 1.5,
        'snr_estimate': 10.0,
        'adaptive_window_size': 5,
        'regularization': 0.01
    }

    if config:
        default_config.update(config)

    wiener_config = WienerConfig(**default_config)
    return WienerWrapper(wiener_config, method)


if __name__ == '__main__':
    # 测试维纳滤波器
    print("Testing Wiener Denoiser...")

    # 创建测试图像（模拟GKP态）
    np.random.seed(42)
    size = 64
    x = np.linspace(-6, 6, size)
    X, Y = np.meshgrid(x, x)

    # 创建GKP样格点
    spacing = 2.5
    ideal = np.zeros((size, size))
    for i in range(-2, 3):
        for j in range(-2, 3):
            ideal += np.exp(-((X - i*spacing)**2 + (Y - j*spacing)**2) / (2 * 0.4**2))
    ideal = ideal / ideal.max()

    # 添加噪声
    noisy = scipy.ndimage.gaussian_filter(ideal, sigma=1.5)
    noisy += np.random.normal(0, 0.02, noisy.shape)
    noisy = noisy / (noisy.max() + 1e-8)

    # 测试各种方法
    config = WienerConfig(kernel_sigma=1.5, snr_estimate=20.0)
    denoiser = WienerDenoiser(config)

    methods = ['wiener', 'wiener_deconv', 'regularized', 'adaptive', 'bilateral']

    print("\nMethod comparison:")
    for method in methods:
        denoised = denoiser.denoise(noisy, method)
        mse = np.mean((denoised - ideal)**2)
        print(f"  {method:20s}: MSE = {mse:.6f}")

    # 测试包装类
    print("\nTesting WienerWrapper...")
    wrapper = WienerWrapper(config, 'adaptive')
    x_tensor = torch.FloatTensor(noisy).unsqueeze(0).unsqueeze(0)
    y_tensor = wrapper(x_tensor)
    print(f"Input shape: {x_tensor.shape}")
    print(f"Output shape: {y_tensor.shape}")
