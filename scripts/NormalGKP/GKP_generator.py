"""
GKP_generator.py - 真实GKP量子态数据集生成器

实现更贴近实际量子纠错场景的GKP码模拟：
1. 有限能量GKP态 (Finite-energy GKP states)
2. 逻辑 |0⟩, |1⟩, |+⟩, |-⟩ 态
3. 多种物理噪声模型：光子损耗、热噪声、位移误差、旋转误差
4. 周期性边界条件
5. 综合征提取与逻辑态标签
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class LogicalState(Enum):
    """GKP逻辑态类型"""
    ZERO = 0      # |0_L⟩
    ONE = 1       # |1_L⟩
    PLUS = 2      # |+_L⟩ = (|0⟩ + |1⟩)/√2
    MINUS = 3     # |-_L⟩ = (|0⟩ - |1⟩)/√2


@dataclass
class NoiseConfig:
    """噪声配置参数"""
    # 扩散噪声 (模拟有限挤压和光子损耗)
    diffusion_sigma: float = 1.5
    diffusion_variance: float = 0.2  # 随机扰动范围

    # 位移误差 (Displacement errors)
    displacement_x_std: float = 0.3  # x方向位移标准差
    displacement_p_std: float = 0.3  # p方向位移标准差

    # 旋转误差 (Phase space rotation)
    rotation_std: float = 0.05  # 旋转角度标准差 (弧度)

    # 加性测量噪声
    measurement_noise_std: float = 0.02

    # 热噪声 (Thermal noise)
    thermal_photon_number: float = 0.1  # 平均热光子数

    # 光子损耗 (Photon loss rate)
    loss_rate: float = 0.1


@dataclass
class GKPConfig:
    """GKP态配置参数"""
    grid_size: int = 64
    x_range: Tuple[float, float] = (-6, 6)

    # GKP格点间距 (理想情况为 √(2π) ≈ 2.507)
    lattice_spacing: float = 2.5

    # 有限能量参数 (Finite energy envelope)
    # 越小表示能量越有限，峰越少
    envelope_sigma: float = 3.0

    # 单个峰的宽度 (与挤压度相关)
    peak_sigma: float = 0.35

    # 相空间范围内的峰数量限制
    max_peaks: int = 9  # 每个方向最多峰数


class GKPStateGenerator:
    """
    GKP量子态生成器

    实现物理上更真实的GKP态模拟，包括：
    - 有限能量包络
    - 正确的逻辑态结构
    - 多种噪声通道
    """

    def __init__(self, config: GKPConfig = None, noise_config: NoiseConfig = None):
        self.config = config or GKPConfig()
        self.noise_config = noise_config or NoiseConfig()

        # 创建相空间网格
        x = np.linspace(self.config.x_range[0], self.config.x_range[1], self.config.grid_size)
        p = np.linspace(self.config.x_range[0], self.config.x_range[1], self.config.grid_size)
        self.X, self.P = np.meshgrid(x, p)
        self.dx = x[1] - x[0]
        self.dp = p[1] - p[0]

        # 格点中心位置
        n_peaks = self.config.max_peaks
        self.peak_indices = np.arange(-(n_peaks//2), n_peaks//2 + 1)

    def _gaussian_2d(self, x0: float, p0: float, sigma: float) -> np.ndarray:
        """生成以(x0, p0)为中心的2D高斯分布"""
        return np.exp(-((self.X - x0)**2 + (self.P - p0)**2) / (2 * sigma**2))

    def _envelope(self, x0: float, p0: float) -> float:
        """有限能量包络函数 - 控制远离原点的峰的衰减"""
        return np.exp(-(x0**2 + p0**2) / (2 * self.config.envelope_sigma**2))

    def generate_ideal_state(self, logical_state: LogicalState) -> np.ndarray:
        """
        生成理想GKP逻辑态的Wigner函数

        GKP码的逻辑态定义：
        |0_L⟩: 峰位于 x = 2n√π (n为整数)
        |1_L⟩: 峰位于 x = (2n+1)√π
        |+_L⟩: 峰位于 p = 2m√π
        |-_L⟩: 峰位于 p = (2m+1)√π

        Wigner函数中，这些表现为相空间中的格点结构
        """
        wigner = np.zeros_like(self.X)
        spacing = self.config.lattice_spacing
        sigma = self.config.peak_sigma

        if logical_state == LogicalState.ZERO:
            # |0_L⟩: x方向偶数格点
            for nx in self.peak_indices:
                for np_ in self.peak_indices:
                    x0 = 2 * nx * spacing / 2  # 偶数位置
                    p0 = np_ * spacing
                    amplitude = self._envelope(x0, p0)
                    if amplitude > 0.01:  # 截断小贡献
                        wigner += amplitude * self._gaussian_2d(x0, p0, sigma)

        elif logical_state == LogicalState.ONE:
            # |1_L⟩: x方向奇数格点
            for nx in self.peak_indices:
                for np_ in self.peak_indices:
                    x0 = (2 * nx + 1) * spacing / 2  # 奇数位置
                    p0 = np_ * spacing
                    amplitude = self._envelope(x0, p0)
                    if amplitude > 0.01:
                        wigner += amplitude * self._gaussian_2d(x0, p0, sigma)

        elif logical_state == LogicalState.PLUS:
            # |+_L⟩: 所有格点，干涉图样在p方向
            for nx in self.peak_indices:
                for np_ in self.peak_indices:
                    x0 = nx * spacing / 2
                    p0 = 2 * np_ * spacing / 2  # p方向偶数
                    amplitude = self._envelope(x0, p0)
                    if amplitude > 0.01:
                        wigner += amplitude * self._gaussian_2d(x0, p0, sigma)

        elif logical_state == LogicalState.MINUS:
            # |-_L⟩: p方向奇数格点
            for nx in self.peak_indices:
                for np_ in self.peak_indices:
                    x0 = nx * spacing / 2
                    p0 = (2 * np_ + 1) * spacing / 2  # p方向奇数
                    amplitude = self._envelope(x0, p0)
                    if amplitude > 0.01:
                        wigner += amplitude * self._gaussian_2d(x0, p0, sigma)

        # 归一化
        if np.max(wigner) > 0:
            wigner = wigner / np.max(wigner)

        return wigner

    def apply_displacement_error(self, wigner: np.ndarray) -> np.ndarray:
        """
        应用位移误差

        物理上对应于相空间中的随机平移
        在数值上通过图像平移实现
        """
        dx_pixels = int(np.random.normal(0, self.noise_config.displacement_x_std) / self.dx)
        dp_pixels = int(np.random.normal(0, self.noise_config.displacement_p_std) / self.dp)

        # 使用周期性边界条件的平移
        wigner = np.roll(wigner, dx_pixels, axis=1)  # x方向
        wigner = np.roll(wigner, dp_pixels, axis=0)  # p方向

        return wigner

    def apply_rotation_error(self, wigner: np.ndarray) -> np.ndarray:
        """
        应用相空间旋转误差

        对应于振荡器频率的微小偏差导致的相位积累
        """
        angle = np.random.normal(0, self.noise_config.rotation_std) * 180 / np.pi  # 转为度

        # 旋转图像（相空间旋转）
        wigner = scipy.ndimage.rotate(wigner, angle, reshape=False, mode='wrap')

        return wigner

    def apply_diffusion_noise(self, wigner: np.ndarray) -> np.ndarray:
        """
        应用扩散噪声（高斯模糊）

        物理上对应于：
        - 有限挤压效应
        - 光子损耗导致的相干性丢失
        - 热噪声导致的相空间扩散
        """
        sigma = self.noise_config.diffusion_sigma
        sigma *= np.random.uniform(
            1 - self.noise_config.diffusion_variance,
            1 + self.noise_config.diffusion_variance
        )

        wigner = scipy.ndimage.gaussian_filter(wigner, sigma=sigma, mode='wrap')

        return wigner

    def apply_thermal_noise(self, wigner: np.ndarray) -> np.ndarray:
        """
        应用热噪声

        热态的Wigner函数是一个宽高斯分布
        热噪声会在整个相空间添加背景
        """
        n_th = self.noise_config.thermal_photon_number
        if n_th > 0:
            # 热态的Wigner函数宽度与热光子数成正比
            thermal_sigma = np.sqrt(n_th + 0.5) * 2
            thermal_background = np.exp(-(self.X**2 + self.P**2) / (2 * thermal_sigma**2))
            thermal_background = thermal_background / np.max(thermal_background) * n_th * 0.1
            wigner = wigner + thermal_background

        return wigner

    def apply_photon_loss(self, wigner: np.ndarray) -> np.ndarray:
        """
        应用光子损耗

        光子损耗导致相空间向原点收缩
        W(x,p) -> W(√η x, √η p) 并伴随高斯卷积
        """
        eta = 1 - self.noise_config.loss_rate
        target_size = self.config.grid_size

        if eta < 1:
            # 缩放相空间（向原点收缩）
            scale = np.sqrt(eta)
            wigner_scaled = scipy.ndimage.zoom(wigner, scale, mode='nearest')

            # 确保输出尺寸正确
            old_h, old_w = wigner_scaled.shape

            # 创建目标大小的数组
            result = np.zeros((target_size, target_size))

            if old_h < target_size:
                # 需要填充：将缩放后的图像居中放置
                pad_h = (target_size - old_h) // 2
                pad_w = (target_size - old_w) // 2
                result[pad_h:pad_h+old_h, pad_w:pad_w+old_w] = wigner_scaled
            else:
                # 需要裁剪：从中心裁剪
                start_h = (old_h - target_size) // 2
                start_w = (old_w - target_size) // 2
                result = wigner_scaled[start_h:start_h+target_size,
                                       start_w:start_w+target_size]

            wigner = result

            # 伴随的高斯卷积（模拟真空噪声）
            loss_blur = np.sqrt(1 - eta) * 0.5
            if loss_blur > 0.1:
                wigner = scipy.ndimage.gaussian_filter(wigner, sigma=loss_blur, mode='wrap')

        return wigner

    def apply_measurement_noise(self, wigner: np.ndarray) -> np.ndarray:
        """应用加性测量噪声"""
        noise = np.random.normal(0, self.noise_config.measurement_noise_std, wigner.shape)
        return wigner + noise

    def apply_all_noise(self, wigner: np.ndarray,
                        apply_displacement: bool = True,
                        apply_rotation: bool = True,
                        apply_diffusion: bool = True,
                        apply_thermal: bool = True,
                        apply_loss: bool = True,
                        apply_measurement: bool = True) -> np.ndarray:
        """
        应用所有噪声通道

        噪声应用顺序模拟物理过程：
        1. 光子损耗（传输过程）
        2. 热噪声（环境耦合）
        3. 位移误差（外部场扰动）
        4. 旋转误差（频率偏移）
        5. 扩散噪声（退相干）
        6. 测量噪声（探测器噪声）
        """
        if apply_loss:
            wigner = self.apply_photon_loss(wigner)
        if apply_thermal:
            wigner = self.apply_thermal_noise(wigner)
        if apply_displacement:
            wigner = self.apply_displacement_error(wigner)
        if apply_rotation:
            wigner = self.apply_rotation_error(wigner)
        if apply_diffusion:
            wigner = self.apply_diffusion_noise(wigner)
        if apply_measurement:
            wigner = self.apply_measurement_noise(wigner)

        return wigner

    def extract_syndrome(self, wigner: np.ndarray) -> Tuple[float, float]:
        """
        提取综合征信息

        GKP码的综合征对应于相空间中峰位置相对于理想格点的偏移
        通过质心计算估计位移
        """
        # 找到主峰位置
        threshold = 0.5 * np.max(wigner)
        mask = wigner > threshold

        if np.sum(mask) == 0:
            return 0.0, 0.0

        # 计算加权质心
        total = np.sum(wigner[mask])
        x_centroid = np.sum(self.X[mask] * wigner[mask]) / total
        p_centroid = np.sum(self.P[mask] * wigner[mask]) / total

        # 计算相对于最近格点的偏移
        spacing = self.config.lattice_spacing
        sx = x_centroid - round(x_centroid / spacing) * spacing
        sp = p_centroid - round(p_centroid / spacing) * spacing

        return sx, sp

    def generate_sample(self,
                       logical_state: Optional[LogicalState] = None,
                       return_syndrome: bool = False) -> Dict:
        """
        生成单个训练样本

        返回：
        - noisy: 噪声Wigner函数
        - ideal: 理想Wigner函数
        - logical_state: 逻辑态标签
        - syndrome: 综合征（可选）
        """
        if logical_state is None:
            logical_state = np.random.choice(list(LogicalState))

        # 生成理想态
        ideal = self.generate_ideal_state(logical_state)

        # 添加随机微扰增加多样性
        shift_x = np.random.uniform(-0.15, 0.15)
        shift_p = np.random.uniform(-0.15, 0.15)
        ideal_shifted = np.roll(ideal, int(shift_x / self.dx), axis=1)
        ideal_shifted = np.roll(ideal_shifted, int(shift_p / self.dp), axis=0)

        # 应用噪声
        noisy = self.apply_all_noise(ideal_shifted.copy())

        # 归一化
        if np.max(noisy) > 0:
            noisy = noisy / (np.max(np.abs(noisy)) + 1e-8)
        if np.max(ideal_shifted) > 0:
            ideal_shifted = ideal_shifted / np.max(ideal_shifted)

        result = {
            'noisy': noisy,
            'ideal': ideal_shifted,
            'logical_state': logical_state.value
        }

        if return_syndrome:
            sx, sp = self.extract_syndrome(noisy)
            result['syndrome'] = (sx, sp)

        return result

    def generate_batch(self, batch_size: int,
                      balanced: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成批量数据

        Args:
            batch_size: 批次大小
            balanced: 是否平衡各逻辑态的数量

        Returns:
            X_noisy: (B, 1, H, W) 噪声输入
            Y_ideal: (B, 1, H, W) 理想输出
            labels: (B,) 逻辑态标签
        """
        noisy_list = []
        ideal_list = []
        label_list = []

        if balanced:
            states = list(LogicalState)
            samples_per_state = batch_size // len(states)
            remainder = batch_size % len(states)

            for i, state in enumerate(states):
                n_samples = samples_per_state + (1 if i < remainder else 0)
                for _ in range(n_samples):
                    sample = self.generate_sample(logical_state=state)
                    noisy_list.append(sample['noisy'])
                    ideal_list.append(sample['ideal'])
                    label_list.append(sample['logical_state'])
        else:
            for _ in range(batch_size):
                sample = self.generate_sample()
                noisy_list.append(sample['noisy'])
                ideal_list.append(sample['ideal'])
                label_list.append(sample['logical_state'])

        # 打乱顺序
        indices = np.random.permutation(len(noisy_list))
        noisy_list = [noisy_list[i] for i in indices]
        ideal_list = [ideal_list[i] for i in indices]
        label_list = [label_list[i] for i in indices]

        X_noisy = torch.FloatTensor(np.array(noisy_list)).unsqueeze(1)
        Y_ideal = torch.FloatTensor(np.array(ideal_list)).unsqueeze(1)
        labels = torch.LongTensor(label_list)

        return X_noisy, Y_ideal, labels


class GKPDataset(Dataset):
    """
    GKP码PyTorch数据集

    支持动态生成或预生成数据
    """

    def __init__(self,
                 size: int,
                 gkp_config: GKPConfig = None,
                 noise_config: NoiseConfig = None,
                 pregenerate: bool = True,
                 balanced: bool = True):
        """
        Args:
            size: 数据集大小
            gkp_config: GKP配置
            noise_config: 噪声配置
            pregenerate: 是否预生成所有数据
            balanced: 是否平衡逻辑态
        """
        self.size = size
        self.generator = GKPStateGenerator(gkp_config, noise_config)
        self.pregenerate = pregenerate
        self.balanced = balanced

        if pregenerate:
            print(f"预生成 {size} 个GKP样本...")
            self.X, self.Y, self.labels = self.generator.generate_batch(size, balanced)
            print("数据生成完成。")
        else:
            self.X = None
            self.Y = None
            self.labels = None

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.pregenerate:
            return self.X[idx], self.Y[idx], self.labels[idx]
        else:
            sample = self.generator.generate_sample()
            noisy = torch.FloatTensor(sample['noisy']).unsqueeze(0)
            ideal = torch.FloatTensor(sample['ideal']).unsqueeze(0)
            label = torch.LongTensor([sample['logical_state']])
            return noisy, ideal, label[0]


def create_dataloaders(train_size: int = 2000,
                       test_size: int = 200,
                       batch_size: int = 32,
                       gkp_config: GKPConfig = None,
                       noise_config: NoiseConfig = None,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试DataLoader
    """
    train_dataset = GKPDataset(train_size, gkp_config, noise_config, pregenerate=True)
    test_dataset = GKPDataset(test_size, gkp_config, noise_config, pregenerate=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# ==========================================
# 可视化工具函数
# ==========================================
def visualize_gkp_states(generator: GKPStateGenerator, save_path: str = None):
    """可视化四种逻辑态"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, state in enumerate(LogicalState):
        # 理想态
        ideal = generator.generate_ideal_state(state)
        axes[0, i].imshow(ideal, cmap='viridis', origin='lower',
                         extent=[generator.config.x_range[0], generator.config.x_range[1],
                                generator.config.x_range[0], generator.config.x_range[1]])
        axes[0, i].set_title(f'Ideal |{state.name}⟩')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('p')

        # 噪声态
        sample = generator.generate_sample(state)
        axes[1, i].imshow(sample['noisy'], cmap='viridis', origin='lower',
                         extent=[generator.config.x_range[0], generator.config.x_range[1],
                                generator.config.x_range[0], generator.config.x_range[1]])
        axes[1, i].set_title(f'Noisy |{state.name}⟩')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('p')

    plt.suptitle('GKP Logical States: Ideal (top) vs Noisy (bottom)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 测试数据生成
    print("Testing GKP Generator...")

    gkp_config = GKPConfig(grid_size=64)
    noise_config = NoiseConfig(diffusion_sigma=1.5)

    generator = GKPStateGenerator(gkp_config, noise_config)

    # 生成批量数据
    X, Y, labels = generator.generate_batch(16)
    print(f"Generated batch: X={X.shape}, Y={Y.shape}, labels={labels.shape}")
    print(f"Label distribution: {torch.bincount(labels)}")

    # 可视化
    visualize_gkp_states(generator)
