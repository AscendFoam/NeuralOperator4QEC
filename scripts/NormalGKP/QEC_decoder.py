"""
QEC_decoder.py - GKP量子纠错解码器

实现从Wigner函数到逻辑态的解码过程：
1. 逻辑态分类器
2. 综合征提取
3. 逻辑错误率计算
4. 物理错误率与逻辑错误率关系分析
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from scipy.ndimage import maximum_filter, label
from scipy.special import erfc


class LogicalState(Enum):
    """GKP逻辑态类型"""
    ZERO = 0      # |0_L⟩
    ONE = 1       # |1_L⟩
    PLUS = 2      # |+_L⟩
    MINUS = 3     # |-_L⟩


@dataclass
class QECConfig:
    """QEC解码配置"""
    lattice_spacing: float = 2.5      # GKP格点间距
    grid_size: int = 64               # 网格大小
    x_range: Tuple[float, float] = (-6, 6)  # 相空间范围
    peak_threshold: float = 0.3       # 峰检测阈值
    decision_method: str = 'intensity'  # 决策方法: 'intensity', 'peak_count', 'correlation'


class GKPLogicalDecoder:
    """
    GKP逻辑态解码器

    从Wigner函数推断逻辑量子比特状态
    """

    def __init__(self, config: QECConfig = None):
        self.config = config or QECConfig()

        # 创建坐标网格
        x = np.linspace(self.config.x_range[0], self.config.x_range[1], self.config.grid_size)
        self.x_coords = x
        self.dx = x[1] - x[0]

        # 预计算格点位置
        self._precompute_lattice_positions()

    def _precompute_lattice_positions(self):
        """预计算各逻辑态的理想格点位置"""
        spacing = self.config.lattice_spacing
        x_range = self.config.x_range

        # 计算在范围内的格点
        n_max = int(np.ceil(x_range[1] / (spacing / 2))) + 1

        # |0⟩态: x方向偶数格点
        self.zero_x_positions = []
        for n in range(-n_max, n_max + 1):
            pos = n * spacing
            if x_range[0] <= pos <= x_range[1]:
                self.zero_x_positions.append(pos)

        # |1⟩态: x方向奇数格点
        self.one_x_positions = []
        for n in range(-n_max, n_max + 1):
            pos = (n + 0.5) * spacing
            if x_range[0] <= pos <= x_range[1]:
                self.one_x_positions.append(pos)

        # p方向类似（用于|+⟩和|-⟩）
        self.plus_p_positions = self.zero_x_positions.copy()
        self.minus_p_positions = self.one_x_positions.copy()

    def _pos_to_idx(self, pos: float) -> int:
        """将物理位置转换为数组索引"""
        idx = int((pos - self.config.x_range[0]) / self.dx)
        return np.clip(idx, 0, self.config.grid_size - 1)

    def _integrate_at_positions(self, wigner: np.ndarray,
                                positions: List[float],
                                axis: int,
                                window_size: int = 3) -> float:
        """
        在指定位置积分Wigner函数

        Args:
            wigner: 2D Wigner函数
            positions: 积分位置列表
            axis: 0表示p方向(行), 1表示x方向(列)
            window_size: 积分窗口半宽

        Returns:
            总积分强度
        """
        total_intensity = 0.0

        for pos in positions:
            idx = self._pos_to_idx(pos)
            start = max(0, idx - window_size)
            end = min(self.config.grid_size, idx + window_size + 1)

            if axis == 1:  # x方向
                # 对该x位置的所有p值求和
                total_intensity += np.sum(wigner[:, start:end])
            else:  # p方向
                # 对该p位置的所有x值求和
                total_intensity += np.sum(wigner[start:end, :])

        return total_intensity

    def decode_logical_state_intensity(self, wigner: np.ndarray) -> LogicalState:
        """
        基于积分强度的解码方法

        比较不同格点位置的强度来判断逻辑态
        """
        # 确保wigner是正值（用于强度计算）
        wigner_pos = np.maximum(wigner, 0)

        # 计算x方向偶数/奇数格点的强度
        zero_intensity = self._integrate_at_positions(
            wigner_pos, self.zero_x_positions, axis=1
        )
        one_intensity = self._integrate_at_positions(
            wigner_pos, self.one_x_positions, axis=1
        )

        # 计算p方向偶数/奇数格点的强度
        plus_intensity = self._integrate_at_positions(
            wigner_pos, self.plus_p_positions, axis=0
        )
        minus_intensity = self._integrate_at_positions(
            wigner_pos, self.minus_p_positions, axis=0
        )

        # 判断Z基矢还是X基矢
        z_basis_contrast = abs(zero_intensity - one_intensity) / (zero_intensity + one_intensity + 1e-8)
        x_basis_contrast = abs(plus_intensity - minus_intensity) / (plus_intensity + minus_intensity + 1e-8)

        if z_basis_contrast > x_basis_contrast:
            # Z基矢态 (|0⟩ or |1⟩)
            if zero_intensity > one_intensity:
                return LogicalState.ZERO
            else:
                return LogicalState.ONE
        else:
            # X基矢态 (|+⟩ or |-⟩)
            if plus_intensity > minus_intensity:
                return LogicalState.PLUS
            else:
                return LogicalState.MINUS

    def decode_logical_state_peaks(self, wigner: np.ndarray) -> LogicalState:
        """
        基于峰位置的解码方法

        检测Wigner函数中的峰，根据峰位置判断逻辑态
        """
        threshold = self.config.peak_threshold * np.max(wigner)

        # 找到局部最大值
        local_max = maximum_filter(wigner, size=5)
        peaks = (wigner == local_max) & (wigner > threshold)

        # 获取峰位置
        peak_positions = np.where(peaks)
        if len(peak_positions[0]) == 0:
            return LogicalState.ZERO  # 默认

        # 转换为物理坐标
        p_peaks = self.x_coords[peak_positions[0]]
        x_peaks = self.x_coords[peak_positions[1]]

        # 统计峰与各格点的接近程度
        spacing = self.config.lattice_spacing

        zero_score = 0
        one_score = 0
        plus_score = 0
        minus_score = 0

        for x, p in zip(x_peaks, p_peaks):
            # x方向: 检查是否接近偶数或奇数格点
            x_mod = (x / spacing) % 1
            if x_mod < 0.25 or x_mod > 0.75:
                zero_score += 1
            else:
                one_score += 1

            # p方向: 类似
            p_mod = (p / spacing) % 1
            if p_mod < 0.25 or p_mod > 0.75:
                plus_score += 1
            else:
                minus_score += 1

        # 决策
        z_diff = abs(zero_score - one_score)
        x_diff = abs(plus_score - minus_score)

        if z_diff > x_diff:
            return LogicalState.ZERO if zero_score > one_score else LogicalState.ONE
        else:
            return LogicalState.PLUS if plus_score > minus_score else LogicalState.MINUS

    def decode_logical_state_correlation(self, wigner: np.ndarray,
                                         templates: Dict[LogicalState, np.ndarray]) -> LogicalState:
        """
        基于模板相关的解码方法

        计算与各逻辑态模板的相关性
        """
        best_corr = -np.inf
        best_state = LogicalState.ZERO

        wigner_norm = wigner - np.mean(wigner)
        wigner_std = np.std(wigner_norm) + 1e-8

        for state, template in templates.items():
            template_norm = template - np.mean(template)
            template_std = np.std(template_norm) + 1e-8

            # 归一化互相关
            corr = np.sum(wigner_norm * template_norm) / (wigner_std * template_std * wigner.size)

            if corr > best_corr:
                best_corr = corr
                best_state = state

        return best_state

    def decode(self, wigner: np.ndarray,
               method: str = None,
               templates: Dict[LogicalState, np.ndarray] = None) -> LogicalState:
        """
        解码主接口

        Args:
            wigner: Wigner函数 (H, W)
            method: 解码方法 ('intensity', 'peaks', 'correlation')
            templates: 模板字典（仅correlation方法需要）

        Returns:
            推断的逻辑态
        """
        method = method or self.config.decision_method

        if method == 'intensity':
            return self.decode_logical_state_intensity(wigner)
        elif method == 'peaks':
            return self.decode_logical_state_peaks(wigner)
        elif method == 'correlation':
            if templates is None:
                raise ValueError("Correlation method requires templates")
            return self.decode_logical_state_correlation(wigner, templates)
        else:
            raise ValueError(f"Unknown method: {method}")

    def decode_batch(self, wigners: np.ndarray, method: str = None) -> np.ndarray:
        """
        批量解码

        Args:
            wigners: (B, H, W) 或 (B, 1, H, W)

        Returns:
            (B,) 逻辑态标签数组
        """
        if wigners.ndim == 4:
            wigners = wigners[:, 0]  # 移除通道维度

        labels = np.array([
            self.decode(w, method).value for w in wigners
        ])
        return labels


def compute_logical_error_rate(
    model,
    test_loader,
    decoder: GKPLogicalDecoder,
    device: torch.device,
    method: str = 'intensity',
    is_mwpm: bool = False
) -> Tuple[float, Dict]:
    """
    计算逻辑错误率

    Args:
        model: 去噪模型
        test_loader: 测试数据加载器
        decoder: GKP解码器
        device: 计算设备
        method: 解码方法
        is_mwpm: 是否是MWPM解码器（直接从噪声解码）

    Returns:
        logical_error_rate: 逻辑错误率
        details: 详细统计信息
    """
    model.eval() if hasattr(model, 'eval') else None

    total_samples = 0
    logical_errors = 0
    confusion_matrix = np.zeros((4, 4), dtype=int)

    # 检查是否是MWPM模型（通过检查是否有mwpm_decoder属性或模型名称）
    has_mwpm_decoder = hasattr(model, 'decoder') and hasattr(model.decoder, 'decode_batch')

    with torch.no_grad():
        for noisy, ideal, labels in test_loader:
            noisy = noisy.to(device)
            labels = labels.numpy()

            if is_mwpm or has_mwpm_decoder:
                # MWPM直接从噪声数据解码，不进行态重建
                noisy_np = noisy.cpu().numpy()
                if noisy_np.ndim == 4:
                    noisy_np = noisy_np[:, 0]

                if has_mwpm_decoder:
                    predicted_labels = model.decoder.decode_batch(noisy_np)
                else:
                    # 使用传入的decoder
                    predicted_labels = decoder.decode_batch(noisy_np, method)
            else:
                # 其他模型：先重建再解码
                reconstructed = model(noisy)

                # 转换为numpy
                if isinstance(reconstructed, torch.Tensor):
                    reconstructed = reconstructed.cpu().numpy()

                # 解码每个样本
                predicted_labels = decoder.decode_batch(reconstructed, method)

            # 统计错误
            for true_label, pred_label in zip(labels, predicted_labels):
                confusion_matrix[true_label, pred_label] += 1
                if true_label != pred_label:
                    logical_errors += 1
                total_samples += 1

    logical_error_rate = logical_errors / total_samples

    # 计算每个逻辑态的准确率
    per_class_accuracy = {}
    for i, state in enumerate(LogicalState):
        class_total = confusion_matrix[i].sum()
        if class_total > 0:
            per_class_accuracy[state.name] = confusion_matrix[i, i] / class_total
        else:
            per_class_accuracy[state.name] = 0.0

    details = {
        'total_samples': total_samples,
        'logical_errors': logical_errors,
        'confusion_matrix': confusion_matrix,
        'per_class_accuracy': per_class_accuracy
    }

    return logical_error_rate, details


def theoretical_logical_error_rate(sigma: float, alpha: float = 2.5) -> float:
    """
    理论逻辑错误率（高斯噪声假设）

    对于GKP码，当位移误差超过α/2时发生逻辑错误
    P_L ≈ erfc(α / (2√2 σ))

    Args:
        sigma: 噪声标准差
        alpha: GKP格点间距

    Returns:
        理论逻辑错误率
    """
    return erfc(alpha / (2 * np.sqrt(2) * sigma + 1e-8))


def estimate_physical_error_rate(noise_sigma: float, alpha: float = 2.5) -> float:
    """
    估计物理错误率

    物理错误率定义为单次操作导致错误的概率
    对于GKP码，可以用噪声强度相对于格点间距的比值估计

    Args:
        noise_sigma: 噪声标准差
        alpha: GKP格点间距

    Returns:
        估计的物理错误率
    """
    # 使用高斯误差函数的近似
    # p ≈ 2σ/α 当 σ << α
    p = 2 * noise_sigma / alpha
    return min(p, 1.0)


class QECBenchmark:
    """
    QEC性能基准测试类

    进行不同噪声水平下的逻辑错误率测试
    """

    def __init__(self, models: Dict[str, object], decoder: GKPLogicalDecoder,
                 device: torch.device):
        """
        Args:
            models: 模型字典 {name: model}
            decoder: GKP解码器
            device: 计算设备
        """
        self.models = models
        self.decoder = decoder
        self.device = device
        self.results = {}

    def run_noise_sweep(self,
                       noise_levels: List[float],
                       create_dataloader_fn,
                       samples_per_level: int = 500,
                       batch_size: int = 32) -> Dict:
        """
        运行噪声扫描实验

        Args:
            noise_levels: 噪声强度列表
            create_dataloader_fn: 创建dataloader的函数 (noise_sigma) -> DataLoader
            samples_per_level: 每个噪声水平的样本数
            batch_size: 批次大小

        Returns:
            results: 实验结果字典
        """
        results = {
            'noise_levels': noise_levels,
            'physical_error_rates': [],
            'theoretical_logical_error_rates': [],
            'model_results': {name: [] for name in self.models.keys()}
        }

        for sigma in noise_levels:
            print(f"\nNoise level σ = {sigma:.2f}")

            # 估计物理错误率
            p_physical = estimate_physical_error_rate(sigma)
            results['physical_error_rates'].append(p_physical)

            # 理论逻辑错误率
            p_logical_theory = theoretical_logical_error_rate(sigma)
            results['theoretical_logical_error_rates'].append(p_logical_theory)

            # 创建该噪声水平的测试数据
            test_loader = create_dataloader_fn(sigma, samples_per_level, batch_size)

            # 测试每个模型
            for name, model in self.models.items():
                # 检测是否是MWPM模型
                is_mwpm = (name.upper() == 'MWPM' or
                          hasattr(model, 'decoder') and hasattr(model.decoder, 'decode_batch'))
                p_logical, details = compute_logical_error_rate(
                    model, test_loader, self.decoder, self.device, is_mwpm=is_mwpm
                )
                results['model_results'][name].append(p_logical)
                print(f"  {name}: p_L = {p_logical:.4f}")

        self.results = results
        return results

    def plot_results(self, save_path: str = None):
        """绘制逻辑错误率 vs 物理错误率曲线"""
        import matplotlib.pyplot as plt

        if not self.results:
            raise ValueError("No results to plot. Run run_noise_sweep first.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 图1: p_L vs 噪声强度
        ax1 = axes[0]
        noise_levels = self.results['noise_levels']

        # 理论曲线
        ax1.semilogy(noise_levels, self.results['theoretical_logical_error_rates'],
                    'k--', label='Theoretical (no correction)', linewidth=2)

        # 各模型曲线
        colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
        markers = ['o', 's', '^', 'D', 'v']

        for i, (name, p_L_list) in enumerate(self.results['model_results'].items()):
            ax1.semilogy(noise_levels, p_L_list, linestyle='-',
                        marker=markers[i % len(markers)],
                        color=colors[i % len(colors)],
                        label=name, linewidth=2, markersize=8)

        ax1.set_xlabel('Noise Level (σ)', fontsize=12)
        ax1.set_ylabel('Logical Error Rate (p_L)', fontsize=12)
        ax1.set_title('Logical Error Rate vs Noise Level', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([1e-3, 1.0])

        # 图2: p_L vs 物理错误率
        ax2 = axes[1]
        p_physical = self.results['physical_error_rates']

        ax2.loglog(p_physical, self.results['theoretical_logical_error_rates'],
                  'k--', label='Theoretical (no correction)', linewidth=2)

        for i, (name, p_L_list) in enumerate(self.results['model_results'].items()):
            ax2.loglog(p_physical, p_L_list, linestyle='-',
                      marker=markers[i % len(markers)],
                      color=colors[i % len(colors)],
                      label=name, linewidth=2, markersize=8)

        # 添加阈值线 (p_L = p)
        p_range = np.logspace(-2, 0, 50)
        ax2.loglog(p_range, p_range, 'gray', linestyle=':', label='p_L = p', alpha=0.7)

        ax2.set_xlabel('Physical Error Rate (p)', fontsize=12)
        ax2.set_ylabel('Logical Error Rate (p_L)', fontsize=12)
        ax2.set_title('Logical Error Rate vs Physical Error Rate', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def print_summary(self):
        """打印结果摘要"""
        if not self.results:
            print("No results available.")
            return

        print("\n" + "="*80)
        print("QEC BENCHMARK SUMMARY")
        print("="*80)

        noise_levels = self.results['noise_levels']
        p_physical = self.results['physical_error_rates']

        # 表头
        header = f"{'σ':>6} | {'p_phys':>8} | {'Theory':>10}"
        for name in self.results['model_results'].keys():
            header += f" | {name:>10}"
        print(header)
        print("-"*80)

        # 数据行
        for i, sigma in enumerate(noise_levels):
            row = f"{sigma:>6.2f} | {p_physical[i]:>8.4f} | {self.results['theoretical_logical_error_rates'][i]:>10.4f}"
            for name, p_L_list in self.results['model_results'].items():
                row += f" | {p_L_list[i]:>10.4f}"
            print(row)

        print("-"*80)

        # 计算改进倍数
        print("\nImprovement over theoretical (at highest noise):")
        theory_worst = self.results['theoretical_logical_error_rates'][-1]
        for name, p_L_list in self.results['model_results'].items():
            model_worst = p_L_list[-1]
            if model_worst > 0:
                improvement = theory_worst / model_worst
                print(f"  {name}: {improvement:.2f}x reduction in logical error rate")


def create_ideal_templates(generator) -> Dict[LogicalState, np.ndarray]:
    """
    创建各逻辑态的理想模板

    Args:
        generator: GKPStateGenerator实例

    Returns:
        模板字典
    """
    templates = {}
    for state in LogicalState:
        templates[state] = generator.generate_ideal_state(state)
    return templates


if __name__ == '__main__':
    # 测试解码器
    print("Testing GKP Logical Decoder...")

    config = QECConfig(lattice_spacing=2.5, grid_size=64)
    decoder = GKPLogicalDecoder(config)

    # 创建测试Wigner函数
    x = np.linspace(-6, 6, 64)
    X, P = np.meshgrid(x, x)

    # 模拟|0⟩态
    spacing = 2.5
    wigner_zero = np.zeros((64, 64))
    for n in range(-2, 3):
        for m in range(-2, 3):
            wigner_zero += np.exp(-((X - n*spacing)**2 + (P - m*spacing)**2) / (2*0.35**2))

    result = decoder.decode(wigner_zero, method='intensity')
    print(f"Decoded |0⟩ state as: {result.name}")

    # 模拟|1⟩态
    wigner_one = np.zeros((64, 64))
    for n in range(-2, 3):
        for m in range(-2, 3):
            wigner_one += np.exp(-((X - (n+0.5)*spacing)**2 + (P - m*spacing)**2) / (2*0.35**2))

    result = decoder.decode(wigner_one, method='intensity')
    print(f"Decoded |1⟩ state as: {result.name}")

    # 测试理论逻辑错误率
    print("\nTheoretical logical error rates:")
    for sigma in [0.5, 1.0, 1.5, 2.0, 2.5]:
        p_L = theoretical_logical_error_rate(sigma)
        p_phys = estimate_physical_error_rate(sigma)
        print(f"  σ={sigma:.1f}: p_physical={p_phys:.4f}, p_logical={p_L:.4f}")
