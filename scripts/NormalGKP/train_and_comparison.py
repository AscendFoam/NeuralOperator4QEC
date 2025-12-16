"""
train_and_comparison.py - GKP态去噪模型训练与对比评估

功能：
1. 训练FNO和UNet模型
2. 对比FNO、UNet和Wiener方法
3. 计算多种评估指标：MSE, PSNR, F1-Score, Precision, Recall
4. 测量推理速度
5. 可视化对比结果
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple, List
from dataclasses import dataclass
import json

# 导入自定义模块
from GKP_generator import (
    GKPStateGenerator, GKPDataset, GKPConfig, NoiseConfig,
    LogicalState, create_dataloaders
)
from FNO import FNO2d_Denoiser, create_fno_denoiser
from UNet import UNet_Denoiser, create_unet_denoiser
from Wiener import WienerWrapper, WienerConfig, create_wiener_denoiser
from MWPM import create_mwpm_decoder
from QEC_decoder import (
    GKPLogicalDecoder, QECConfig, QECBenchmark,
    compute_logical_error_rate, theoretical_logical_error_rate,
    estimate_physical_error_rate
)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据集参数
    train_size: int = 3000
    test_size: int = 300
    batch_size: int = 32

    # 训练参数
    epochs: int = 30
    lr_fno: float = 0.002
    lr_unet: float = 0.001
    weight_decay: float = 1e-5

    # 学习率调度
    use_scheduler: bool = True
    scheduler_step: int = 10
    scheduler_gamma: float = 0.5

    # 评估参数
    peak_threshold: float = 0.5
    speed_test_repetitions: int = 100

    # 保存和可视化
    save_models: bool = True
    save_dir: str = './checkpoints'
    visualize_every: int = 5


class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """均方误差"""
        return F.mse_loss(pred, target).item()

    @staticmethod
    def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
        """峰值信噪比"""
        mse = F.mse_loss(pred, target).item()
        if mse < 1e-10:
            return 100.0
        return 10 * np.log10(1.0 / mse)

    @staticmethod
    def calculate_classification_metrics(pred: torch.Tensor, target: torch.Tensor,
                                        threshold: float = 0.5) -> Dict[str, float]:
        """计算分类指标：Precision, Recall, F1"""
        pred_mask = (pred > threshold).float()
        target_mask = (target > threshold).float()

        tp = (pred_mask * target_mask).sum().item()
        fp = (pred_mask * (1 - target_mask)).sum().item()
        fn = ((1 - pred_mask) * target_mask).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    @staticmethod
    def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
        """结构相似性指数（简化版）"""
        # 使用滑动窗口计算局部统计量
        c1, c2 = 0.01**2, 0.03**2

        mu_pred = pred.mean()
        mu_target = target.mean()
        sigma_pred = pred.std()
        sigma_target = target.std()
        sigma_cross = ((pred - mu_pred) * (target - mu_target)).mean()

        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)) / \
               ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred**2 + sigma_target**2 + c2))

        return ssim.item()

    @classmethod
    def calculate_all(cls, pred: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5) -> Dict[str, float]:
        """计算所有指标"""
        metrics = {
            'mse': cls.calculate_mse(pred, target),
            'psnr': cls.calculate_psnr(pred, target),
            'ssim': cls.calculate_ssim(pred, target)
        }
        metrics.update(cls.calculate_classification_metrics(pred, target, threshold))
        return metrics


def measure_inference_speed(model, sample_input: torch.Tensor,
                           repetitions: int = 100,
                           warmup: int = 10) -> float:
    """
    测量模型推理速度

    Returns:
        每样本平均推理时间（毫秒）
    """
    model.eval()
    device = sample_input.device

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)

    # 同步CUDA（如果使用GPU）
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 计时
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(sample_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_per_sample = (total_time / (repetitions * sample_input.shape[0])) * 1000

    return avg_time_per_sample


class Trainer:
    """模型训练器"""

    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        self.metrics_calc = MetricsCalculator()

        # 创建保存目录
        if config.save_models:
            os.makedirs(config.save_dir, exist_ok=True)

    def train_epoch(self, model: nn.Module, optimizer: optim.Optimizer,
                   train_loader: DataLoader, criterion: nn.Module) -> float:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0

        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, model, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y, _ in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                pred = model(batch_x)
                all_preds.append(pred)
                all_targets.append(batch_y)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        return self.metrics_calc.calculate_all(
            all_preds, all_targets, self.config.peak_threshold
        )

    def train_model(self, model: nn.Module, train_loader: DataLoader,
                   test_loader: DataLoader, name: str,
                   lr: float) -> Dict:
        """完整训练流程"""
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"{'='*50}")

        optimizer = optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=self.config.weight_decay)

        scheduler = None
        if self.config.use_scheduler:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.scheduler_step,
                gamma=self.config.scheduler_gamma
            )

        criterion = nn.MSELoss()

        history = {
            'train_loss': [],
            'test_metrics': []
        }

        best_f1 = 0.0

        for epoch in range(self.config.epochs):
            # 训练
            train_loss = self.train_epoch(model, optimizer, train_loader, criterion)
            history['train_loss'].append(train_loss)

            # 评估
            test_metrics = self.evaluate(model, test_loader)
            history['test_metrics'].append(test_metrics)

            # 学习率调度
            if scheduler:
                scheduler.step()

            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.6f} | "
                      f"F1: {test_metrics['f1']*100:.2f}% | "
                      f"PSNR: {test_metrics['psnr']:.2f}dB")

            # 保存最佳模型
            if self.config.save_models and test_metrics['f1'] > best_f1:
                best_f1 = test_metrics['f1']
                torch.save(model.state_dict(),
                          os.path.join(self.config.save_dir, f'{name}_best.pth'))

        return history


def plot_comparison_results(noisy: torch.Tensor, ideal: torch.Tensor,
                           fno_pred: torch.Tensor, unet_pred: torch.Tensor,
                           wiener_pred: torch.Tensor, save_path: str = None):
    """绘制对比结果"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # 第一行：图像对比
    images = [
        noisy[0, 0].cpu().numpy(),
        ideal[0, 0].cpu().numpy(),
        fno_pred[0, 0].detach().cpu().numpy(),
        unet_pred[0, 0].detach().cpu().numpy(),
        wiener_pred[0, 0].cpu().numpy()
    ]
    titles = ['Noisy Input', 'Ideal Target', 'FNO Output', 'UNet Output', 'Wiener Output']

    for i, (img, title) in enumerate(zip(images, titles)):
        im = axes[0, i].imshow(img, cmap='viridis', origin='lower', vmin=0, vmax=1)
        axes[0, i].set_title(title, fontsize=12)
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

    # 第二行：误差图
    errors = [
        np.abs(images[0] - images[1]),  # Noisy error
        np.zeros_like(images[1]),        # Target (no error)
        np.abs(images[2] - images[1]),   # FNO error
        np.abs(images[3] - images[1]),   # UNet error
        np.abs(images[4] - images[1])    # Wiener error
    ]
    error_titles = ['Noise Level', '-', 'FNO Error', 'UNet Error', 'Wiener Error']

    for i, (err, title) in enumerate(zip(errors, error_titles)):
        if i == 1:
            axes[1, i].axis('off')
            continue
        im = axes[1, i].imshow(err, cmap='hot', origin='lower', vmin=0, vmax=0.5)
        axes[1, i].set_title(title, fontsize=12)
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.suptitle('GKP State Reconstruction Comparison', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(fno_history: Dict, unet_history: Dict, save_path: str = None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(fno_history['train_loss']) + 1)

    # 训练损失
    axes[0, 0].plot(epochs, fno_history['train_loss'], 'b-', label='FNO')
    axes[0, 0].plot(epochs, unet_history['train_loss'], 'r-', label='UNet')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # F1 Score
    fno_f1 = [m['f1'] * 100 for m in fno_history['test_metrics']]
    unet_f1 = [m['f1'] * 100 for m in unet_history['test_metrics']]
    axes[0, 1].plot(epochs, fno_f1, 'b-', label='FNO')
    axes[0, 1].plot(epochs, unet_f1, 'r-', label='UNet')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score (%)')
    axes[0, 1].set_title('F1 Score Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # PSNR
    fno_psnr = [m['psnr'] for m in fno_history['test_metrics']]
    unet_psnr = [m['psnr'] for m in unet_history['test_metrics']]
    axes[1, 0].plot(epochs, fno_psnr, 'b-', label='FNO')
    axes[1, 0].plot(epochs, unet_psnr, 'r-', label='UNet')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].set_title('PSNR Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # MSE
    fno_mse = [m['mse'] for m in fno_history['test_metrics']]
    unet_mse = [m['mse'] for m in unet_history['test_metrics']]
    axes[1, 1].semilogy(epochs, fno_mse, 'b-', label='FNO')
    axes[1, 1].semilogy(epochs, unet_mse, 'r-', label='UNet')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE (log)')
    axes[1, 1].set_title('MSE Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_benchmark_table(results: Dict[str, Dict], model_params: Dict[str, int]):
    """打印性能对比表格"""
    print("\n" + "="*100)
    print("FINAL BENCHMARK RESULTS: FNO vs UNet vs Wiener")
    print("="*100)

    header = f"{'Model':<10} | {'Params':>10} | {'MSE':>12} | {'F1-Score':>10} | " \
             f"{'Precision':>10} | {'Recall':>10} | {'PSNR':>8} | {'Speed(ms)':>10}"
    print(header)
    print("-"*100)

    for name, metrics in results.items():
        params = model_params.get(name, 0)
        param_str = f"{params:,}" if params > 0 else "N/A"
        print(f"{name:<10} | {param_str:>10} | {metrics['mse']:>12.8f} | "
              f"{metrics['f1']*100:>9.2f}% | {metrics['precision']*100:>9.2f}% | "
              f"{metrics['recall']*100:>9.2f}% | {metrics['psnr']:>8.2f} | "
              f"{metrics['speed']:>10.3f}")

    print("-"*100)


def analyze_results(results: Dict[str, Dict]):
    """分析并总结结果"""
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)

    # 找出各指标最佳
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    best_psnr = max(results.items(), key=lambda x: x[1]['psnr'])
    best_speed = min(results.items(), key=lambda x: x[1]['speed'])

    print(f"\nBest F1 Score: {best_f1[0]} ({best_f1[1]['f1']*100:.2f}%)")
    print(f"Best PSNR: {best_psnr[0]} ({best_psnr[1]['psnr']:.2f} dB)")
    print(f"Fastest: {best_speed[0]} ({best_speed[1]['speed']:.3f} ms/sample)")

    # 性能对比
    fno = results.get('FNO', {})
    unet = results.get('UNet', {})
    wiener = results.get('Wiener', {})

    if fno and unet:
        if fno['f1'] > unet['f1']:
            improvement = (fno['f1'] - unet['f1']) / unet['f1'] * 100
            print(f"\nFNO F1 improvement over UNet: {improvement:.1f}%")
        else:
            improvement = (unet['f1'] - fno['f1']) / fno['f1'] * 100
            print(f"\nUNet F1 improvement over FNO: {improvement:.1f}%")

        if fno['speed'] < unet['speed']:
            speedup = unet['speed'] / fno['speed']
            print(f"FNO is {speedup:.2f}x faster than UNet")
        else:
            speedup = fno['speed'] / unet['speed']
            print(f"UNet is {speedup:.2f}x faster than FNO")

    if fno and wiener:
        if fno['f1'] > wiener['f1']:
            improvement = (fno['f1'] - wiener['f1']) / (wiener['f1'] + 1e-8) * 100
            print(f"FNO F1 improvement over Wiener: {improvement:.1f}%")

    # 量子纠错相关分析
    print("\n--- Quantum Error Correction Implications ---")
    if fno.get('f1', 0) > 0.95:
        print("[Success] FNO achieves >95% F1, suitable for high-fidelity state reconstruction")
    if fno.get('speed', float('inf')) < 1.0:
        print("[Speed] Sub-millisecond inference enables real-time QEC feedback")


def run_experiment():
    """运行完整实验"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 配置
    training_config = TrainingConfig(
        train_size=3000,
        test_size=300,
        batch_size=32,
        epochs=30
    )

    gkp_config = GKPConfig(
        grid_size=64,
        lattice_spacing=2.5,
        envelope_sigma=3.0,
        peak_sigma=0.35
    )

    noise_config = NoiseConfig(
        diffusion_sigma=1.8,
        displacement_x_std=0.25,
        displacement_p_std=0.25,
        rotation_std=0.03,
        measurement_noise_std=0.02,
        thermal_photon_number=0.05,
        loss_rate=0.08
    )

    wiener_config = WienerConfig(
        kernel_sigma=noise_config.diffusion_sigma,
        snr_estimate=15.0,
        adaptive_window_size=5
    )

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Grid Size: {gkp_config.grid_size}x{gkp_config.grid_size}")
    print(f"Noise Level (diffusion): {noise_config.diffusion_sigma}")

    # 创建数据加载器
    print("\nGenerating datasets...")
    train_loader, test_loader = create_dataloaders(
        train_size=training_config.train_size,
        test_size=training_config.test_size,
        batch_size=training_config.batch_size,
        gkp_config=gkp_config,
        noise_config=noise_config
    )

    # 创建模型
    print("\nInitializing models...")
    fno_model = create_fno_denoiser({
        'modes': 12,
        'width': 32,
        'n_layers': 4,
        'use_coord_embed': True
    }).to(device)

    unet_model = create_unet_denoiser({
        'base_channels': 32,
        'depth': 4,
        'use_attention': False
    }).to(device)

    wiener_model = create_wiener_denoiser(
        {'kernel_sigma': wiener_config.kernel_sigma},
        method='adaptive'
    )

    model_params = {
        'FNO': fno_model.count_parameters(),
        'UNet': unet_model.count_parameters(),
        'Wiener': 0
    }

    print(f"FNO parameters: {model_params['FNO']:,}")
    print(f"UNet parameters: {model_params['UNet']:,}")

    # 训练
    trainer = Trainer(training_config, device)

    fno_history = trainer.train_model(
        fno_model, train_loader, test_loader,
        'FNO', training_config.lr_fno
    )

    unet_history = trainer.train_model(
        unet_model, train_loader, test_loader,
        'UNet', training_config.lr_unet
    )

    # 绘制训练曲线
    plot_training_history(fno_history, unet_history, 'training_history.png')

    # 最终评估
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)

    # 获取测试数据
    test_x, test_y = [], []
    for bx, by, _ in test_loader:
        test_x.append(bx)
        test_y.append(by)
    test_x = torch.cat(test_x, dim=0).to(device)
    test_y = torch.cat(test_y, dim=0).to(device)

    # 推理
    fno_model.eval()
    unet_model.eval()

    with torch.no_grad():
        fno_pred = fno_model(test_x)
        unet_pred = unet_model(test_x)
        wiener_pred = wiener_model(test_x)

    # 计算指标
    metrics_calc = MetricsCalculator()

    results = {
        'FNO': metrics_calc.calculate_all(fno_pred, test_y, training_config.peak_threshold),
        'UNet': metrics_calc.calculate_all(unet_pred, test_y, training_config.peak_threshold),
        'Wiener': metrics_calc.calculate_all(wiener_pred, test_y, training_config.peak_threshold)
    }

    # 测速
    speed_sample = test_x[:training_config.batch_size]

    results['FNO']['speed'] = measure_inference_speed(
        fno_model, speed_sample, training_config.speed_test_repetitions
    )
    results['UNet']['speed'] = measure_inference_speed(
        unet_model, speed_sample, training_config.speed_test_repetitions
    )

    # Wiener测速（CPU）
    t0 = time.perf_counter()
    for _ in range(10):
        _ = wiener_model(test_x[:1].cpu())
    results['Wiener']['speed'] = (time.perf_counter() - t0) / 10 * 1000

    # 打印结果表格
    print_benchmark_table(results, model_params)

    # 分析结果
    analyze_results(results)

    # 可视化对比
    sample_x = test_x[:4]
    sample_y = test_y[:4]

    with torch.no_grad():
        sample_fno = fno_model(sample_x)
        sample_unet = unet_model(sample_x)
        sample_wiener = wiener_model(sample_x)

    plot_comparison_results(
        sample_x, sample_y,
        sample_fno, sample_unet, sample_wiener,
        'comparison_results.png'
    )

    # 保存结果
    results_to_save = {
        name: {k: float(v) for k, v in metrics.items()}
        for name, metrics in results.items()
    }
    with open('benchmark_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print("\nExperiment completed!")
    print("Results saved to benchmark_results.json")

    # 创建MWPM解码器
    mwpm_model = create_mwpm_decoder({
        'lattice_spacing': gkp_config.lattice_spacing,
        'grid_size': gkp_config.grid_size,
        'x_range': gkp_config.x_range
    })

    return results, fno_model, unet_model, wiener_model, mwpm_model, gkp_config, noise_config


def run_qec_evaluation(models: Dict, gkp_config: GKPConfig, base_noise_config: NoiseConfig,
                       device: torch.device):
    """
    运行QEC解码评估

    在不同物理错误率下测试逻辑错误率
    """
    print("\n" + "="*70)
    print("QEC DECODING EVALUATION")
    print("="*70)

    # 创建QEC解码器
    qec_config = QECConfig(
        lattice_spacing=gkp_config.lattice_spacing,
        grid_size=gkp_config.grid_size,
        x_range=gkp_config.x_range
    )
    decoder = GKPLogicalDecoder(qec_config)

    # 噪声水平扫描范围
    noise_levels = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]

    def create_test_loader(noise_sigma: float, n_samples: int = 400, batch_size: int = 32):
        """创建指定噪声水平的测试数据加载器"""
        noise_config = NoiseConfig(
            diffusion_sigma=noise_sigma,
            displacement_x_std=base_noise_config.displacement_x_std,
            displacement_p_std=base_noise_config.displacement_p_std,
            rotation_std=base_noise_config.rotation_std,
            measurement_noise_std=base_noise_config.measurement_noise_std,
            thermal_photon_number=base_noise_config.thermal_photon_number,
            loss_rate=base_noise_config.loss_rate
        )
        dataset = GKPDataset(n_samples, gkp_config, noise_config, pregenerate=True, balanced=True)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 运行基准测试
    benchmark = QECBenchmark(models, decoder, device)
    results = benchmark.run_noise_sweep(
        noise_levels=noise_levels,
        create_dataloader_fn=create_test_loader,
        samples_per_level=400,
        batch_size=32
    )

    # 打印摘要
    benchmark.print_summary()

    # 绘制结果
    benchmark.plot_results(save_path='qec_logical_error_rate.png')

    return results


def plot_combined_results(reconstruction_results: Dict, qec_results: Dict, save_path: str = None):
    """绘制综合结果图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 颜色方案 (支持4个模型)
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

    # 图1: 重建质量对比 (F1 Score) - 只显示有重建能力的模型
    ax1 = axes[0, 0]
    reconstruction_models = [m for m in reconstruction_results.keys() if m != 'MWPM']
    f1_scores = [reconstruction_results[m]['f1'] * 100 for m in reconstruction_models]
    bars = ax1.bar(reconstruction_models, f1_scores, color=colors[:len(reconstruction_models)])
    ax1.set_ylabel('F1 Score (%)', fontsize=12)
    ax1.set_title('State Reconstruction Quality (F1 Score)', fontsize=14)
    ax1.set_ylim([0, 105])
    for bar, score in zip(bars, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.1f}%', ha='center', va='bottom', fontsize=11)

    # 图2: 重建质量对比 (PSNR) - 只显示有重建能力的模型
    ax2 = axes[0, 1]
    psnr_scores = [reconstruction_results[m]['psnr'] for m in reconstruction_models]
    bars = ax2.bar(reconstruction_models, psnr_scores, color=colors[:len(reconstruction_models)])
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('State Reconstruction Quality (PSNR)', fontsize=14)
    for bar, score in zip(bars, psnr_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{score:.1f}', ha='center', va='bottom', fontsize=11)

    # 图3: 逻辑错误率 vs 噪声
    ax3 = axes[1, 0]
    noise_levels = qec_results['noise_levels']

    ax3.semilogy(noise_levels, qec_results['theoretical_logical_error_rates'],
                'k--', label='No Correction', linewidth=2)

    markers = ['o', 's', '^', 'D']  # 支持4个模型
    for i, (name, p_L_list) in enumerate(qec_results['model_results'].items()):
        ax3.semilogy(noise_levels, p_L_list, linestyle='-',
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=name, linewidth=2, markersize=8)

    ax3.set_xlabel('Noise Level (σ)', fontsize=12)
    ax3.set_ylabel('Logical Error Rate (p_L)', fontsize=12)
    ax3.set_title('QEC Performance: Logical Error Rate', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([1e-3, 1.0])

    # 图4: 逻辑错误率 vs 物理错误率
    ax4 = axes[1, 1]
    p_physical = qec_results['physical_error_rates']

    ax4.loglog(p_physical, qec_results['theoretical_logical_error_rates'],
              'k--', label='No Correction', linewidth=2)

    for i, (name, p_L_list) in enumerate(qec_results['model_results'].items()):
        ax4.loglog(p_physical, p_L_list, linestyle='-',
                  marker=markers[i % len(markers)],
                  color=colors[i % len(colors)],
                  label=name, linewidth=2, markersize=8)

    # 添加 p_L = p 参考线
    p_range = np.logspace(-1.5, 0, 50)
    ax4.loglog(p_range, p_range, 'gray', linestyle=':', label='p_L = p', alpha=0.7)

    ax4.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax4.set_ylabel('Logical Error Rate (p_L)', fontsize=12)
    ax4.set_title('Logical vs Physical Error Rate', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('GKP Quantum Error Correction: Neural Network vs Traditional Methods',
                fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Combined results saved to {save_path}")

    plt.show()


def print_qec_analysis(reconstruction_results: Dict, qec_results: Dict):
    """打印QEC分析总结"""
    print("\n" + "="*70)
    print("COMPREHENSIVE QEC ANALYSIS")
    print("="*70)

    print("\n1. State Reconstruction Performance:")
    print("-"*50)
    for name, metrics in reconstruction_results.items():
        print(f"   {name:10s}: F1={metrics['f1']*100:5.2f}%, PSNR={metrics['psnr']:.2f}dB, "
              f"MSE={metrics['mse']:.2e}")

    print("\n2. QEC Logical Error Rate (at σ=2.0):")
    print("-"*50)
    idx_2 = qec_results['noise_levels'].index(2.0) if 2.0 in qec_results['noise_levels'] else -3
    theory_pL = qec_results['theoretical_logical_error_rates'][idx_2]
    print(f"   Theoretical (no correction): {theory_pL:.4f}")

    for name, p_L_list in qec_results['model_results'].items():
        p_L = p_L_list[idx_2]
        improvement = theory_pL / (p_L + 1e-8)
        print(f"   {name:10s}: p_L={p_L:.4f} ({improvement:.1f}x improvement)")

    print("\n3. Error Suppression Analysis:")
    print("-"*50)
    # 在低噪声区域的表现
    low_noise_idx = 0
    low_noise = qec_results['noise_levels'][low_noise_idx]
    print(f"   At low noise (σ={low_noise}):")
    for name, p_L_list in qec_results['model_results'].items():
        p_L = p_L_list[low_noise_idx]
        if p_L < 0.01:
            print(f"      {name}: p_L < 1% - Excellent error suppression")
        elif p_L < 0.05:
            print(f"      {name}: p_L = {p_L*100:.2f}% - Good error suppression")
        else:
            print(f"      {name}: p_L = {p_L*100:.2f}% - Moderate performance")

    # 在高噪声区域的表现
    high_noise_idx = -1
    high_noise = qec_results['noise_levels'][high_noise_idx]
    print(f"\n   At high noise (σ={high_noise}):")
    for name, p_L_list in qec_results['model_results'].items():
        p_L = p_L_list[high_noise_idx]
        theory = qec_results['theoretical_logical_error_rates'][high_noise_idx]
        ratio = (theory - p_L) / theory * 100
        print(f"      {name}: {ratio:.1f}% reduction in logical error rate")

    print("\n4. Recommendations:")
    print("-"*50)
    # 找出最佳模型
    best_model = min(qec_results['model_results'].items(),
                     key=lambda x: sum(x[1]) / len(x[1]))
    print(f"   Best overall QEC performance: {best_model[0]}")

    # 分析各模型优势
    for name, p_L_list in qec_results['model_results'].items():
        avg_pL = sum(p_L_list) / len(p_L_list)
        if avg_pL < 0.1:
            print(f"   {name}: Suitable for high-fidelity quantum computing")
        elif avg_pL < 0.3:
            print(f"   {name}: Suitable for fault-tolerant protocols with concatenation")
        else:
            print(f"   {name}: May require additional error mitigation")


def run_full_experiment():
    """运行完整实验：训练 + 重建评估 + QEC评估"""
    # 运行基础实验
    results = run_experiment()
    reconstruction_results, fno_model, unet_model, wiener_model, mwpm_model, gkp_config, noise_config = results

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备模型字典 (包含MWPM)
    models = {
        'FNO': fno_model,
        'UNet': unet_model,
        'Wiener': wiener_model,
        'MWPM': mwpm_model
    }

    # MWPM不进行态重建，为其添加占位指标
    reconstruction_results['MWPM'] = {
        'mse': float('nan'),
        'psnr': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'ssim': 0.0,
        'speed': 0.0
    }

    # 运行QEC评估
    qec_results = run_qec_evaluation(models, gkp_config, noise_config, device)

    # 绘制综合结果
    plot_combined_results(reconstruction_results, qec_results, 'full_qec_comparison.png')

    # 打印分析
    print_qec_analysis(reconstruction_results, qec_results)

    # 保存完整结果
    full_results = {
        'reconstruction': {
            name: {k: float(v) for k, v in metrics.items()}
            for name, metrics in reconstruction_results.items()
        },
        'qec': {
            'noise_levels': qec_results['noise_levels'],
            'physical_error_rates': qec_results['physical_error_rates'],
            'theoretical_logical_error_rates': qec_results['theoretical_logical_error_rates'],
            'model_results': qec_results['model_results']
        }
    }
    with open('full_qec_results.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "="*70)
    print("FULL EXPERIMENT COMPLETED")
    print("="*70)
    print("Saved files:")
    print("  - benchmark_results.json (reconstruction metrics)")
    print("  - full_qec_results.json (complete QEC analysis)")
    print("  - comparison_results.png (visual comparison)")
    print("  - qec_logical_error_rate.png (p_L curves)")
    print("  - full_qec_comparison.png (combined analysis)")

    return reconstruction_results, qec_results


if __name__ == '__main__':
    run_full_experiment()
