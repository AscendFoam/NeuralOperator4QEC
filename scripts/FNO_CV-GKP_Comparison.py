import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. CV 物理模拟器: Wigner 函数生成与加噪
# ==========================================
class CVSimulator:
    """
    模拟连续变量(CV)量子态的维格纳函数(Wigner Function)。
    这里模拟一个近似的 GKP 态：相空间中一系列整齐排列的高斯峰。
    噪声表现为扩散过程（高斯模糊）。
    """
    def __init__(self, grid_size=64, x_range=(-6, 6)):
        self.grid_size = grid_size
        # 创建相空间网格 (x, p)
        x = np.linspace(x_range[0], x_range[1], grid_size)
        p = np.linspace(x_range[0], x_range[1], grid_size)
        self.X, self.P = np.meshgrid(x, p)
        self.grid_shape = self.X.shape

    def _gaussian_2d(self, x0, p0, sigma=0.5):
        """生成一个以 (x0, p0) 为中心的二维高斯峰"""
        return np.exp(-((self.X - x0)**2 + (self.P - p0)**2) / (2 * sigma**2))

    def generate_batch(self, batch_size, noise_level=1.0):
        """
        生成一批数据：
        Input: Noisy Wigner Function (模糊的图像)
        Target: Ideal Wigner Function (清晰的图像)
        """
        ideal_batch = []
        noisy_batch = []

        # GKP 峰的间隔 (sqrt(pi) 的倍数，这里简化取整以便可视化)
        spacing = 3.0 
        # 在网格范围内生成峰的中心点
        centers = np.arange(-3, 4) * spacing

        for _ in range(batch_size):
            # 1. 生成理想态 (Ideal State)
            # GKP 态是多个高斯峰的叠加
            wigner_ideal = np.zeros(self.grid_shape)
            
            # 随机选择是一个逻辑 |0> 态（在偶数位置有峰）还是逻辑 |1> 态（在奇数位置有峰）
            # 这里为了简化去噪任务，我们生成一个通用的格点态，不区分逻辑0/1，重点在于恢复结构
            # 如果要区分，只需移动中心点即可。
            
            # 随机微扰一下中心点位置，增加数据多样性
            shift_x = np.random.uniform(-0.2, 0.2)
            shift_p = np.random.uniform(-0.2, 0.2)
            
            for cx in centers:
                for cp in centers:
                    # 只在网格范围内生成
                    if abs(cx) < 5 and abs(cp) < 5:
                        wigner_ideal += self._gaussian_2d(cx + shift_x, cp + shift_p, sigma=0.4)
            
            # 归一化 (为了NN训练稳定)
            wigner_ideal = wigner_ideal / np.max(wigner_ideal)
            ideal_batch.append(wigner_ideal)

            # 2. 添加噪声 (Diffusion Noise)
            # 物理上的扩散过程对应 Wigner 函数的高斯模糊
            # noise_level 控制模糊核的大小
            blur_sigma = noise_level * np.random.uniform(0.8, 1.2) # 增加一点随机性
            wigner_noisy = scipy.ndimage.gaussian_filter(wigner_ideal, sigma=blur_sigma)
            
            # 简单的加性白噪声 (模拟测量噪声)
            wigner_noisy += np.random.normal(0, 0.02, self.grid_shape)
            
            # 重新归一化
            wigner_noisy = wigner_noisy / (np.max(wigner_noisy) + 1e-8)
            noisy_batch.append(wigner_noisy)

        # 转换为 Tensor, Shape: (Batch, 1, H, W)
        X_noisy = torch.FloatTensor(np.array(noisy_batch)).unsqueeze(1)
        Y_ideal = torch.FloatTensor(np.array(ideal_batch)).unsqueeze(1)
        
        return X_noisy, Y_ideal

# ==========================================
# 2. FNO 模型: 用于图像去噪 (Image-to-Image)
# ==========================================
class SpectralConv2d(nn.Module):
    """标准的 2D 傅里叶层"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        current_modes1 = min(self.modes1, x_ft.size(-2))
        current_modes2 = min(self.modes2, x_ft.size(-1))

        out_ft[:, :, :current_modes1, :current_modes2] = \
            self.compl_mul2d(x_ft[:, :, :current_modes1, :current_modes2], self.weights1[:, :, :current_modes1, :current_modes2])
        out_ft[:, :, -current_modes1:, :current_modes2] = \
            self.compl_mul2d(x_ft[:, :, -current_modes1:, :current_modes2], self.weights2[:, :, :current_modes1, :current_modes2])

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d_Denoiser(nn.Module):
    """
    用于 CV 态去噪的 FNO 模型。
    特点：输入和输出都是 2D 图像，没有全局池化层。
    """
    def __init__(self, modes, width):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # 输入是 1 个通道 (Wigner function value)
        self.p = nn.Conv2d(1, self.width, 1) 
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Projection back to image (Pixel-wise regression)
        self.q1 = nn.Conv2d(self.width, 64, 1)
        self.q2 = nn.Conv2d(64, 1, 1) # 输出 1 个通道的图像

    def forward(self, x):
        x = self.p(x)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        
        # 不做 Global Pooling，保持空间结构
        x = F.gelu(self.q1(x))
        x = self.q2(x) 
        # Wigner 函数通常在 0 附近波动，用线性输出或 Tanh 比较合适
        # 这里因为模拟数据归一化到了 [0, 1]，用 Sigmoid 也可以保证输出在范围内
        return torch.sigmoid(x) 

# ==========================================
# 2.1 新增对比模型: U-Net (CNN基准)
# ==========================================
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet_Denoiser(nn.Module):
    """
    简化版 U-Net，用于图像去噪对比。
    使用 Circular Padding 适应边界条件。
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet_Denoiser, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1
        
        # Encoder (Downsampling)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128 // factor))
        
        # Decoder (Upsampling)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(128, 64 // factor)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2) # Bottleneck
        
        x = self.up1(x3)
        # Skip connection (concat)
        x = torch.cat([x2, x], dim=1) 
        x = self.conv_up1(x)
        
        x = self.up2(x)
        # Skip connection (concat)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up2(x)
        
        logits = self.outc(x)
        return torch.sigmoid(logits) # 输出归一化到 [0,1]

# ==========================================
# 2.2 新增对比基准: 维纳滤波 (传统物理/信号方法)
# ==========================================
def wiener_deconvolution_batch(noisy_batch_np, kernel_sigma, snr_guess=10.0):
    """
    对一个批次的图像进行维纳去卷积。
    需要估计扩散核的大小 (kernel_sigma) 和信噪比 (snr_guess)。
    """
    batch_size, _, h, w = noisy_batch_np.shape
    denoised_batch = np.zeros_like(noisy_batch_np)
    
    # 构建估计的高斯扩散核 (PSF)
    # 核大小需要足够大以覆盖扩散范围
    k_size = int(kernel_sigma * 4) * 2 + 1
    xv, yv = np.meshgrid(np.arange(-k_size//2+1, k_size//2+1), np.arange(-k_size//2+1, k_size//2+1))
    psf_kernel = np.exp(-(xv**2 + yv**2) / (2 * kernel_sigma**2))
    psf_kernel /= psf_kernel.sum() # 归一化
    
    for i in range(batch_size):
        img = noisy_batch_np[i, 0]
        # 使用 Wiener 滤波反卷积
        # balance 参数调节平滑度，类似 1/SNR
        denoised_img = scipy.signal.wiener(img, mysize=5, noise=1.0/snr_guess)
        
        # 注：scipy.signal.wiener 是自适应去噪，不是严格的逆卷积。
        # 严格的维纳逆卷积通常在频域做：F_hat = F_noisy * conj(H) / (|H|^2 + 1/SNR)
        # 但在噪声未知时，scipy 的空间域近似实现通常是一个足够强的 baseline。
        # 为了更物理，我们这里用一个简单的锐化算子代替复杂的频域维纳，
        # 或者就用 scipy.signal.wiener 作为通用去噪基准。
        # 这里暂用 scipy.signal.wiener 演示流程。
        
        denoised_batch[i, 0] = denoised_img
        
    return torch.FloatTensor(denoised_batch)

def calculate_metrics(pred_tensor, target_tensor, threshold=0.5):
    """
    计算更公正的分类指标：Precision, Recall, F1-Score
    以及 MSE, PSNR
    """
    mse = F.mse_loss(pred_tensor, target_tensor).item()

    # 转换为 0/1 的二值掩码
    pred_mask = (pred_tensor > threshold).float()
    target_mask = (target_tensor > threshold).float()

    # --- 混淆矩阵计算 ---
    # TP (True Positive): 预测是峰，实际也是峰
    tp = (pred_mask * target_mask).sum().item()
    # FP (False Positive): 预测是峰，实际是背景
    fp = (pred_mask * (1 - target_mask)).sum().item()
    # FN (False Negative): 预测是背景，实际是峰
    fn = ((1 - pred_mask) * target_mask).sum().item()

    # 1. Precision (精准率) = TP / (TP + FP)
    precision = tp / (tp + fp + 1e-8)

    # 2. Recall (召回率) = TP / (TP + FN)
    recall = tp / (tp + fn + 1e-8)

    # 3. F1-Score (综合指标)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # PSNR
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10 * np.log10(1.0 / mse)

    return mse, precision, recall, f1, psnr

def measure_inference_speed(model, sample_input, repetitions=100):
    """测算推理耗时 (毫秒/样本)"""
    model.eval()
    # 预热
    with torch.no_grad():
        _ = model(sample_input)
    
    # 计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repetitions):
            _ = model(sample_input)
    end_time = time.time()
    
    total_time = end_time - start_time
    # 计算每个样本的平均耗时 (ms)
    avg_time_per_sample = (total_time / (repetitions * sample_input.shape[0])) * 1000
    return avg_time_per_sample

# ==========================================
# 3. 修改辅助函数：可视化对比
# ==========================================
def plot_comparison(noisy_t, ideal_t, fno_t, unet_t, wiener_t, epoch):
    """绘制五张对比图"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # 取 batch 中的第一个样本，转为 numpy
    imgs = [
        noisy_t[0, 0].cpu().numpy(),
        ideal_t[0, 0].cpu().numpy(),
        fno_t[0, 0].detach().cpu().numpy(),
        unet_t[0, 0].detach().cpu().numpy(),
        wiener_t[0, 0].numpy() # Wiener 已经是 numpy 转过来的 cpu tensor
    ]
    titles = ["Noisy Input", "Ideal Target", "FNO Output", "UNet Output", "Wiener Baseline"]
    
    vmin, vmax = 0, 1
    
    for ax, img, title in zip(axes, imgs, titles):
        im = ax.imshow(img, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
    
    plt.suptitle(f"CV State Reconstruction Comparison - Epoch {epoch}")
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. 修改实验主流程
# ==========================================
def run_cv_comparison_experiment():
    # 配置
    GRID_SIZE = 64      
    TRAIN_SIZE = 2000
    TEST_SIZE = 100     # 用于计算最终指标
    BATCH_SIZE = 32
    EPOCHS = 20 # U-Net 收敛可能慢一点
    NOISE_LEVEL = 2.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running comparison on {device}. Grid Size: {GRID_SIZE}x{GRID_SIZE}, Noise: {NOISE_LEVEL}")

    # 1. 数据准备
    sim = CVSimulator(grid_size=GRID_SIZE)
    print("Generating training data...")
    X_train, Y_train = sim.generate_batch(TRAIN_SIZE, noise_level=NOISE_LEVEL)
    train_loader = DataLoader(TensorDataset(X_train.to(device), Y_train.to(device)), 
                              batch_size=BATCH_SIZE, shuffle=True)
    
    print("Generating test data...")
    X_test, Y_test = sim.generate_batch(TEST_SIZE, noise_level=NOISE_LEVEL)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    # 2. 模型构建
    # FNO (参数与之前类似)
    fno = FNO2d_Denoiser(modes=12, width=32).to(device)
    # UNet (通道数调整到参数量与 FNO 大致在同一数量级)
    unet = UNet_Denoiser(n_channels=1, n_classes=1).to(device)
    
    opt_fno = optim.AdamW(fno.parameters(), lr=0.002, weight_decay=1e-5)
    opt_unet = optim.AdamW(unet.parameters(), lr=0.001, weight_decay=1e-5) # UNet通常LR小点
    criterion = nn.MSELoss()

    # 3. 训练循环 (同时训练 FNO 和 UNet)
    print("\nStarting Dual Training (FNO vs UNet)...")
    for epoch in range(EPOCHS):
        fno.train(); unet.train()
        loss_f = 0.0; loss_u = 0.0
        
        for bx, by in train_loader:
            # Train FNO
            opt_fno.zero_grad()
            pred_f = fno(bx)
            lf = criterion(pred_f, by)
            lf.backward()
            opt_fno.step()
            loss_f += lf.item()
            
            # Train UNet
            opt_unet.zero_grad()
            pred_u = unet(bx)
            lu = criterion(pred_u, by)
            lu.backward()
            opt_unet.step()
            loss_u += lu.item()
            
        avg_lf = loss_f / len(train_loader)
        avg_lu = loss_u / len(train_loader)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1} | FNO Loss: {avg_lf:.6f} | UNet Loss: {avg_lu:.6f}")
            
            # 可视化对比
            fno.eval(); unet.eval()
            with torch.no_grad():
                # 取测试集前几个样本用于可视化
                vis_x = X_test[:2]; vis_y = Y_test[:2]
                
                pred_f_vis = fno(vis_x)
                pred_u_vis = unet(vis_x)
                # 计算 Wiener 基准 (在 CPU 上计算 Numpy)
                pred_w_vis = wiener_deconvolution_batch(vis_x.cpu().numpy(), kernel_sigma=NOISE_LEVEL)
                
                plot_comparison(vis_x, vis_y, pred_f_vis, pred_u_vis, pred_w_vis, epoch+1)

    # ==========================================
    # 4. 最终全方位评估 (Accuracy, PSNR, Speed)
    # ==========================================
    print("\n" + "="*50)
    print("FINAL BENCHMARK: FNO vs UNet vs Wiener")
    print("="*50)
    
    fno.eval(); unet.eval()
    
    # 准备数据
    X_test_cpu = X_test.cpu().numpy()
    Y_test_cpu = Y_test.cpu()
    
    # --- 1. 批量推理获取结果 ---
    with torch.no_grad():
        # FNO
        pred_f = fno(X_test)
        # UNet
        pred_u = unet(X_test)
        # Wiener (CPU)
        pred_w_np = wiener_deconvolution_batch(X_test_cpu, kernel_sigma=NOISE_LEVEL)
        pred_w = pred_w_np.to(device)

    # --- 2. 计算各项指标 ---
    # 定义阈值，用于判定是否是一个有效的 GKP 峰
    PEAK_THRESHOLD = 0.5

    mse_f, prec_f, rec_f, f1_f, psnr_f = calculate_metrics(pred_f, Y_test, PEAK_THRESHOLD)
    mse_u, prec_u, rec_u, f1_u, psnr_u = calculate_metrics(pred_u, Y_test, PEAK_THRESHOLD)
    mse_w, prec_w, rec_w, f1_w, psnr_w = calculate_metrics(pred_w, Y_test, PEAK_THRESHOLD)

    # --- 3. 测速 (Latency) ---
    # 取一个 batch 进行测速
    speed_sample = X_test[:BATCH_SIZE]
    time_f = measure_inference_speed(fno, speed_sample)
    time_u = measure_inference_speed(unet, speed_sample)
    # Wiener 速度通常很慢且在 CPU，这里简单估算单次
    t0 = time.time()
    wiener_deconvolution_batch(X_test_cpu[:1], kernel_sigma=NOISE_LEVEL)
    time_w = (time.time() - t0) * 1000 # ms

    # --- 4. 打印报表 ---
    print(f"{'Model':<8} | {'MSE':>12} | {'F1-Score':>10} | {'Precision':>10} | {'Recall':>10} | {'PSNR':>8} | {'Speed(ms)':>10}")
    print("-" * 90)
    print(f"{'FNO':<8} | {mse_f:>12.8f} | {f1_f*100:>9.2f}% | {prec_f*100:>9.2f}% | {rec_f*100:>9.2f}% | {psnr_f:>8.2f} | {time_f:>10.3f}")
    print(f"{'UNet':<8} | {mse_u:>12.8f} | {f1_u*100:>9.2f}% | {prec_u*100:>9.2f}% | {rec_u*100:>9.2f}% | {psnr_u:>8.2f} | {time_u:>10.3f}")
    print(f"{'Wiener':<8} | {mse_w:>12.8f} | {f1_w*100:>9.2f}% | {prec_w*100:>9.2f}% | {rec_w*100:>9.2f}% | {psnr_w:>8.2f} | {time_w:>10.3f}")
    print("-" * 90)

    # --- 5. 结论分析 ---
    print("\nAnalysis:")
    if f1_f > 0.99:
        print(" [Success] FNO achieved near-perfect reconstruction of quantum states.")
    if time_f < time_u:
        print(f" [Speed] FNO is {time_u/time_f:.1f}x faster than UNet (Crucial for real-time QEC).")
    else:
        print(f" [Speed] FNO is comparable to UNet.")
        
    if psnr_f > psnr_u + 5:
        print(" [Quality] FNO signal quality is significantly higher (>5dB boost).")

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    run_cv_comparison_experiment()
    