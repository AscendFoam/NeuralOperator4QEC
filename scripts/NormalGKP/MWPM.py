"""
MWPM.py - Analog Minimum Weight Perfect Matching 解码器用于GKP码

实现GKP码的模拟信息MWPM解码：
1. 从Wigner函数提取模拟综合征（位移信息）
2. 基于位移的软判决解码
3. 简化的MWPM算法用于GKP逻辑态判决

参考文献：
- Terhal & Weigand, "Encoding a qubit into a oscillator", PRX 2020
- Royer et al., "Stabilization and operation of a GKP qubit", Nature 2020
"""

import numpy as np
import torch
from scipy.ndimage import maximum_filter, center_of_mass
from scipy.optimize import linear_sum_assignment
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class MWPMConfig:
    """MWPM解码器配置"""
    lattice_spacing: float = 2.5      # GKP格点间距
    grid_size: int = 64               # 网格大小
    x_range: Tuple[float, float] = (-6, 6)
    peak_threshold: float = 0.3       # 峰检测阈值
    use_soft_decision: bool = True    # 使用软判决
    sigma_prior: float = 1.0          # 先验噪声估计


class AnalogMWPMDecoder:
    """
    GKP码的模拟MWPM解码器

    核心思想：
    1. 提取相空间中峰的位置（模拟综合征）
    2. 计算每个峰相对于理想GKP格点的位移
    3. 使用加权匹配或似然估计进行逻辑态判决

    对于GKP码，MWPM的"匹配"体现在：
    - 将检测到的峰与理想格点位置匹配
    - 根据位移计算似然，决定逻辑态
    """

    def __init__(self, config: MWPMConfig = None):
        self.config = config or MWPMConfig()

        # 创建坐标网格
        x = np.linspace(self.config.x_range[0], self.config.x_range[1],
                       self.config.grid_size)
        self.x_coords = x
        self.dx = x[1] - x[0]

        # 预计算理想格点位置
        self._compute_ideal_lattice()

    def _compute_ideal_lattice(self):
        """预计算理想GKP格点位置"""
        spacing = self.config.lattice_spacing
        x_range = self.config.x_range

        # 计算范围内的所有格点
        n_max = int(np.ceil(x_range[1] / spacing)) + 1

        # 所有格点位置 (用于|0⟩和|1⟩区分)
        self.even_lattice_x = []  # |0⟩态: x = n * spacing (n为整数)
        self.odd_lattice_x = []   # |1⟩态: x = (n + 0.5) * spacing

        for n in range(-n_max, n_max + 1):
            even_pos = n * spacing
            odd_pos = (n + 0.5) * spacing

            if x_range[0] <= even_pos <= x_range[1]:
                self.even_lattice_x.append(even_pos)
            if x_range[0] <= odd_pos <= x_range[1]:
                self.odd_lattice_x.append(odd_pos)

        # p方向类似
        self.even_lattice_p = self.even_lattice_x.copy()
        self.odd_lattice_p = self.odd_lattice_x.copy()

    def _detect_peaks(self, wigner: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        检测Wigner函数中的峰

        Returns:
            List of (x, p, intensity) tuples
        """
        threshold = self.config.peak_threshold * np.max(wigner)

        # 局部最大值检测
        local_max = maximum_filter(wigner, size=5)
        peaks_mask = (wigner == local_max) & (wigner > threshold)

        # 获取峰位置和强度
        peak_indices = np.where(peaks_mask)
        peaks = []

        for i, j in zip(peak_indices[0], peak_indices[1]):
            p_val = self.x_coords[i]
            x_val = self.x_coords[j]
            intensity = wigner[i, j]
            peaks.append((x_val, p_val, intensity))

        return peaks

    def _compute_displacement_to_lattice(self, pos: float,
                                         lattice: List[float]) -> Tuple[float, float]:
        """
        计算位置到最近格点的位移

        Returns:
            (nearest_lattice_point, displacement)
        """
        if not lattice:
            return 0.0, pos

        distances = [abs(pos - lp) for lp in lattice]
        min_idx = np.argmin(distances)
        nearest = lattice[min_idx]
        displacement = pos - nearest

        return nearest, displacement

    def _compute_syndrome(self, wigner: np.ndarray) -> Dict:
        """
        从Wigner函数提取模拟综合征

        综合征包括：
        - x方向的平均位移
        - p方向的平均位移
        - 峰分布统计
        """
        peaks = self._detect_peaks(wigner)

        if not peaks:
            # 无峰检测到，使用质心
            wigner_pos = np.maximum(wigner, 0)
            total = np.sum(wigner_pos)
            if total > 0:
                X, P = np.meshgrid(self.x_coords, self.x_coords)
                x_centroid = np.sum(X * wigner_pos) / total
                p_centroid = np.sum(P * wigner_pos) / total
                peaks = [(x_centroid, p_centroid, 1.0)]
            else:
                return {'x_displacement': 0, 'p_displacement': 0,
                       'x_score_even': 0, 'x_score_odd': 0,
                       'p_score_even': 0, 'p_score_odd': 0}

        # 计算加权位移
        total_intensity = sum(p[2] for p in peaks)

        x_displacements_even = []
        x_displacements_odd = []
        p_displacements_even = []
        p_displacements_odd = []

        for x, p, intensity in peaks:
            weight = intensity / (total_intensity + 1e-8)

            # x方向位移
            _, disp_even_x = self._compute_displacement_to_lattice(x, self.even_lattice_x)
            _, disp_odd_x = self._compute_displacement_to_lattice(x, self.odd_lattice_x)

            x_displacements_even.append((disp_even_x, weight))
            x_displacements_odd.append((disp_odd_x, weight))

            # p方向位移
            _, disp_even_p = self._compute_displacement_to_lattice(p, self.even_lattice_p)
            _, disp_odd_p = self._compute_displacement_to_lattice(p, self.odd_lattice_p)

            p_displacements_even.append((disp_even_p, weight))
            p_displacements_odd.append((disp_odd_p, weight))

        # 计算加权平均位移的平方（越小越好）
        x_score_even = sum(d**2 * w for d, w in x_displacements_even)
        x_score_odd = sum(d**2 * w for d, w in x_displacements_odd)
        p_score_even = sum(d**2 * w for d, w in p_displacements_even)
        p_score_odd = sum(d**2 * w for d, w in p_displacements_odd)

        return {
            'x_score_even': x_score_even,
            'x_score_odd': x_score_odd,
            'p_score_even': p_score_even,
            'p_score_odd': p_score_odd,
            'n_peaks': len(peaks)
        }

    def _likelihood_ratio(self, score_0: float, score_1: float) -> float:
        """
        计算似然比

        基于高斯噪声模型：P(data|state) ∝ exp(-score / (2σ²))
        """
        sigma2 = self.config.sigma_prior ** 2
        # Log likelihood ratio
        llr = (score_1 - score_0) / (2 * sigma2 + 1e-8)
        return llr

    def decode(self, wigner: np.ndarray) -> int:
        """
        解码单个Wigner函数

        返回逻辑态标签：
        0: |0⟩, 1: |1⟩, 2: |+⟩, 3: |-⟩
        """
        syndrome = self._compute_syndrome(wigner)

        # Z基矢判决 (|0⟩ vs |1⟩)
        llr_z = self._likelihood_ratio(
            syndrome['x_score_even'],
            syndrome['x_score_odd']
        )

        # X基矢判决 (|+⟩ vs |-⟩)
        llr_x = self._likelihood_ratio(
            syndrome['p_score_even'],
            syndrome['p_score_odd']
        )

        # 判断是Z基矢态还是X基矢态
        # 比较哪个基矢的匹配更好
        z_min_score = min(syndrome['x_score_even'], syndrome['x_score_odd'])
        x_min_score = min(syndrome['p_score_even'], syndrome['p_score_odd'])

        if z_min_score < x_min_score:
            # Z基矢态
            if llr_z > 0:
                return 0  # |0⟩
            else:
                return 1  # |1⟩
        else:
            # X基矢态
            if llr_x > 0:
                return 2  # |+⟩
            else:
                return 3  # |-⟩

    def decode_batch(self, wigners: np.ndarray) -> np.ndarray:
        """批量解码"""
        if wigners.ndim == 4:
            wigners = wigners[:, 0]

        labels = np.array([self.decode(w) for w in wigners])
        return labels


class MWPMWrapper:
    """
    MWPM解码器包装类

    提供类似神经网络的接口
    注意：MWPM是直接解码器，不进行态重建
    """

    def __init__(self, config: MWPMConfig = None):
        self.config = config or MWPMConfig()
        self.decoder = AnalogMWPMDecoder(self.config)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播接口

        注意：MWPM不进行态重建，这里返回原始输入
        实际的解码在QEC评估阶段进行
        """
        # MWPM不修改Wigner函数，直接返回输入
        # 逻辑态判决在QEC_decoder中进行
        return x

    def eval(self):
        return self

    def train(self):
        return self

    def parameters(self):
        return []

    def count_parameters(self) -> int:
        return 0


class MWPMDirectDecoder:
    """
    直接MWPM解码器

    绕过态重建，直接从噪声Wigner函数解码逻辑态
    """

    def __init__(self, config: MWPMConfig = None):
        self.config = config or MWPMConfig()
        self.mwpm_decoder = AnalogMWPMDecoder(self.config)

    def decode_from_wigner(self, wigner: np.ndarray) -> int:
        """从Wigner函数直接解码"""
        return self.mwpm_decoder.decode(wigner)

    def decode_batch_from_wigner(self, wigners: np.ndarray) -> np.ndarray:
        """批量解码"""
        return self.mwpm_decoder.decode_batch(wigners)


def compute_mwpm_logical_error_rate(
    test_loader,
    mwpm_config: MWPMConfig,
    device: torch.device
) -> Tuple[float, Dict]:
    """
    计算MWPM的逻辑错误率

    MWPM直接从噪声数据解码，不需要预处理
    """
    decoder = AnalogMWPMDecoder(mwpm_config)

    total_samples = 0
    logical_errors = 0
    confusion_matrix = np.zeros((4, 4), dtype=int)

    for noisy, ideal, labels in test_loader:
        noisy_np = noisy.numpy()
        if noisy_np.ndim == 4:
            noisy_np = noisy_np[:, 0]

        labels_np = labels.numpy()

        # MWPM直接解码
        predicted_labels = decoder.decode_batch(noisy_np)

        for true_label, pred_label in zip(labels_np, predicted_labels):
            confusion_matrix[true_label, pred_label] += 1
            if true_label != pred_label:
                logical_errors += 1
            total_samples += 1

    logical_error_rate = logical_errors / total_samples

    details = {
        'total_samples': total_samples,
        'logical_errors': logical_errors,
        'confusion_matrix': confusion_matrix
    }

    return logical_error_rate, details


def create_mwpm_decoder(config: dict = None) -> MWPMWrapper:
    """
    工厂函数：创建MWPM解码器

    Args:
        config: 配置字典

    Returns:
        MWPMWrapper 实例
    """
    default_config = {
        'lattice_spacing': 2.5,
        'grid_size': 64,
        'x_range': (-6, 6),
        'peak_threshold': 0.3,
        'use_soft_decision': True,
        'sigma_prior': 1.0
    }

    if config:
        default_config.update(config)

    mwpm_config = MWPMConfig(**default_config)
    return MWPMWrapper(mwpm_config)


if __name__ == '__main__':
    # 测试MWPM解码器
    print("Testing Analog MWPM Decoder...")

    config = MWPMConfig(lattice_spacing=2.5, grid_size=64)
    decoder = AnalogMWPMDecoder(config)

    # 创建测试Wigner函数
    x = np.linspace(-6, 6, 64)
    X, P = np.meshgrid(x, x)
    spacing = 2.5

    print("\nTesting logical state decoding:")

    # |0⟩态测试
    wigner_zero = np.zeros((64, 64))
    for n in range(-2, 3):
        for m in range(-2, 3):
            wigner_zero += np.exp(-((X - n*spacing)**2 + (P - m*spacing)**2) / (2*0.35**2))
    result = decoder.decode(wigner_zero)
    print(f"  |0⟩ state decoded as: {result} (expected: 0)")

    # |1⟩态测试
    wigner_one = np.zeros((64, 64))
    for n in range(-2, 3):
        for m in range(-2, 3):
            wigner_one += np.exp(-((X - (n+0.5)*spacing)**2 + (P - m*spacing)**2) / (2*0.35**2))
    result = decoder.decode(wigner_one)
    print(f"  |1⟩ state decoded as: {result} (expected: 1)")

    # |+⟩态测试
    wigner_plus = np.zeros((64, 64))
    for n in range(-2, 3):
        for m in range(-2, 3):
            wigner_plus += np.exp(-((X - n*spacing/2)**2 + (P - m*spacing)**2) / (2*0.35**2))
    result = decoder.decode(wigner_plus)
    print(f"  |+⟩ state decoded as: {result} (expected: 2)")

    # 添加噪声测试
    print("\nTesting with noise:")
    import scipy.ndimage
    noisy_zero = scipy.ndimage.gaussian_filter(wigner_zero, sigma=1.5)
    noisy_zero += np.random.normal(0, 0.02, noisy_zero.shape)
    result = decoder.decode(noisy_zero)
    print(f"  Noisy |0⟩ state decoded as: {result}")

    print("\nMWPM decoder test completed.")
