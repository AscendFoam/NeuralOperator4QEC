import os
# Fix OpenMP duplicate library issue on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import copy

# ==========================================
# 1. 物理模拟器: GKP Toric Code Simulator
# ==========================================
class GKPToricCodeSim:
    """
    模拟 L x L 的 Toric Code 在 GKP 唯象噪声下的行为。
    为了简化，我们模拟两个独立的网格：一个用于 X 校验（检测 Z 错误），一个用于 Z 校验（检测 X 错误）。
    """
    def __init__(self, L):
        self.L = L
        self.n_qubits = 2 * L * L
        self.sqrt_pi = np.sqrt(np.pi)
        
    def generate_batch(self, batch_size, sigma_noise):
        """
        生成数据:
        Input: (Batch, 2, L, L, 4) -> 包含 X 和 Z 稳定子的 sin/cos 模拟值
        Label: (Batch, 2) -> 逻辑 X 错误和 逻辑 Z 错误 (0 或 1)
        """
        # 1. 生成物理噪声 (Displacement) ~ N(0, sigma^2)
        # 形状: [Batch, 2, L, L]。这里简化模型：
        # channel 0: 垂直边上的噪声 (主要影响 Z 稳定子)
        # channel 1: 水平边上的噪声
        noise = np.random.normal(0, sigma_noise, (batch_size, 2, self.L, self.L))
        
        # 2. 提取症候群 (Syndrome)
        # Toric Code 的稳定子是周围 4 个边的噪声之和
        # 模 sqrt(pi) 是 GKP 的特征
        
        # 为了演示 FNO 的强大，我们构建一个简化的 Syndrome 映射：
        # Syndrome(i, j) = Noise_up + Noise_down + Noise_left + Noise_right
        # 注意处理周期性边界 (np.roll)
        
        # Z-Stabilizers (检测 X 错误)
        # 假设 noise 是 X-basis 的位移
        s_z = (noise[:, 0, :, :] + 
               np.roll(noise[:, 0, :, :], -1, axis=1) + # shift x
               noise[:, 1, :, :] + 
               np.roll(noise[:, 1, :, :], -1, axis=2))  # shift y
        
        # 模拟 GKP 测量: 加上测量噪声并模 sqrt(pi)
        # 此处省略测量噪声以聚焦核心逻辑，直接取模
        s_z_analog = (s_z + self.sqrt_pi/2) % self.sqrt_pi - self.sqrt_pi/2
        
        # 3. 计算逻辑错误 (Ground Truth)
        # 在 Toric Code 中，如果一行的净位移超过 sqrt(pi)/2，则发生逻辑错误
        # 逻辑算符 Z_L 是穿过环面的一条线上的 X 错误之和
        logical_err_1 = np.sum(noise[:, 0, :, 0], axis=1) # 沿 y 轴积分
        logical_err_2 = np.sum(noise[:, 1, 0, :], axis=1) # 沿 x 轴积分
        
        # 判断是否翻转 (GKP 逻辑: 积分值接近 sqrt(pi) 的奇数倍)
        # 简单判据: round(sum / sqrt(pi)) % 2 != 0
        y_1 = np.round(logical_err_1 / self.sqrt_pi).astype(int) % 2
        y_2 = np.round(logical_err_2 / self.sqrt_pi).astype(int) % 2
        
        # 4. 特征工程: 周期性嵌入 [sin, cos]
        # 输入给神经网络的是 Analog Syndrome
        # 我们对数据做归一化 scale 到 [0, 2pi] 以便 sin/cos 编码
        scale = 2 * np.pi / self.sqrt_pi
        feat_sin = np.sin(s_z_analog * scale)
        feat_cos = np.cos(s_z_analog * scale)
        
        # 堆叠特征: (Batch, 2, L, L) -> 2 channels: sin, cos
        # 这里只演示 Z-stabilizer 解码逻辑 X 错误
        X = np.stack([feat_sin, feat_cos], axis=1) 
        Y = y_1 # 这是一个二分类问题
        
        return torch.FloatTensor(X), torch.FloatTensor(Y).unsqueeze(1)

# ==========================================
# 2. 模型定义: CNN Baseline
# ==========================================
class CNNBaseline(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        # 使用 Circular Padding 适配 Toric Code 拓扑
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, padding_mode='circular')
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

# ==========================================
# 3. 模型定义: FNO Challenger (Resolution Invariant)
# ==========================================
class SpectralConv2d(nn.Module):
    """
    2D Fourier Layer. 
    关键点：权重定义在低频模式上，高频自动补零/截断，从而实现分辨率无关。
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Fourier modes to keep in x
        self.modes2 = modes2 # Fourier modes to keep in y

        scale = (1 / (in_channels * out_channels))
        # 权重是复数张量
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # 1. 2D FFT
        # rfft2 输出形状: (batch, c, x, y/2 + 1)
        x_ft = torch.fft.rfft2(x)

        # 2. 频域滤波 (Multiply relevant modes)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # 处理 Corner Cases (如果是小尺寸，modes可能比输入尺寸大，需要截断)
        m1 = min(self.modes1, x_ft.size(-2) // 2)
        m2 = min(self.modes2, x_ft.size(-1))

        # 上左角频率
        out_ft[:, :, :m1, :m2] = \
            self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        # 下左角频率 (负频率)
        out_ft[:, :, -m1:, :m2] = \
            self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        # 3. 2D IFFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes, width):
        super(FNO2d, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        
        # --- 关键修改 1: Local Feature Extraction ---
        # 原来是 1x1 Conv，现在改为 3x3 Conv + Circular Padding
        # 这让 FNO 能够先“看清”局部的错误，再进行全局传输
        self.p = nn.Conv2d(2, self.width, 3, padding=1, padding_mode='circular') 
        
        # --- 关键修改 2: 增加深度 ---
        # 2层 Fourier Layer 对于复杂的纠错逻辑可能不够，增加到 4 层
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2) # 新增
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2) # 新增
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1) # 新增
        self.w3 = nn.Conv2d(self.width, self.width, 1) # 新增

        # Projection
        self.q1 = nn.Linear(self.width, 128)
        self.q2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.p(x) # Local feature extraction
        
        # Layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        # Layer 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        # Layer 2 (New)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)
        
        # Layer 3 (New)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)
        
        # Output: 回到 Global Average Pooling
        # 对于 L 变化的场景，Mean 是最鲁棒的，前提是特征提取做得好
        x = x.mean(dim=(2, 3)) 
        
        x = F.gelu(self.q1(x))
        x = self.q2(x)
        return torch.sigmoid(x)

# ==========================================
# 4. 实验流程控制
# ==========================================
def run_experiment():
    # 配置
    TRAIN_L = 5
    TEST_L_LIST = [5, 7, 9] # 泛化测试
    TRAIN_SIZE = 10000
    TEST_SIZE = 2000
    EPOCHS = 20 # 快速验证，不需要太多轮
    BATCH_SIZE = 64
    SIGMA_TRAIN = 0.26 # 对应某个 dB (约 9dB)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 1. 准备训练数据 (Only d=5)
    print(f"\nGenerating Training Data (L={TRAIN_L}, sigma={SIGMA_TRAIN})...")
    sim_train = GKPToricCodeSim(TRAIN_L)
    x_train, y_train = sim_train.generate_batch(TRAIN_SIZE, SIGMA_TRAIN)
    train_loader = DataLoader(TensorDataset(x_train.to(device), y_train.to(device)), batch_size=BATCH_SIZE, shuffle=True)

    # 2. 初始化模型
    # 1. 提升通道数 (Width) 以承载更多信息
    # 2. Modes 保持为 3 (对于 L=5 来说，modes=3 几乎覆盖了全频段，这是对的)
    fno = FNO2d(modes=3, width=64).to(device) 
    
    # CNN 保持不变，作为基准
    cnn = CNNBaseline().to(device)
    
    # 3. 稍微降低学习率，因为模型变深了
    opt_fno = optim.AdamW(fno.parameters(), lr=0.001, weight_decay=1e-4)
    opt_cnn = optim.AdamW(cnn.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # 3. 训练循环
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        fno.train()
        cnn.train()
        loss_f = 0.0
        loss_c = 0.0
        
        for bx, by in train_loader:
            # FNO Update
            opt_fno.zero_grad()
            pred_f = fno(bx)
            l_f = criterion(pred_f, by)
            l_f.backward()
            opt_fno.step()
            loss_f += l_f.detach().item()

            # CNN Update
            opt_cnn.zero_grad()
            pred_c = cnn(bx)
            l_c = criterion(pred_c, by)
            l_c.backward()
            opt_cnn.step()
            loss_c += l_c.detach().item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | FNO Loss: {loss_f/len(train_loader):.4f} | CNN Loss: {loss_c/len(train_loader):.4f}")

    # 4. 泛化性评估 (Zero-shot Super-Resolution)
    print("\nStarting Evaluation (Zero-Shot Generalization)...")
    results = {'L': [], 'FNO': [], 'CNN': [], 'MWPM_Proxy': []}
    
    fno.eval()
    cnn.eval()
    
    with torch.no_grad():
        for L_test in TEST_L_LIST:
            # 在 L_test 上生成测试数据
            sim_test = GKPToricCodeSim(L_test)
            x_test, y_test = sim_test.generate_batch(TEST_SIZE, SIGMA_TRAIN)
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            # --- FNO 推理 ---
            # FNO 的神奇之处：虽然训练是 5x5，但代码里的 rfft2 会自动适应 L_test 的尺寸
            # 低频权重会自动应用到左上角
            pred_f = fno(x_test)
            acc_f = ((pred_f > 0.5) == y_test).float().mean().item()
            
            # --- CNN 推理 ---
            # CNN 可以处理变长输入 (全是 Conv)，但 Global Pooling 会把空间信息压扁
            # 它能否泛化取决于它学到的是局部特征还是全局位置特征
            pred_c = cnn(x_test)
            acc_c = ((pred_c > 0.5) == y_test).float().mean().item()
            
            # --- Analog MWPM Proxy (Baseline) ---
            # 这里简单模拟 MWPM 的行为：如果总位移小，通常能解对。
            # 为了公平对比，我们只看物理噪声水平。实际 MWPM 会有一个理论阈值。
            # 这里我们假设一个理论上的“好解码器”在同等噪声下，随着 L 增加，错误率应该指数下降。
            # 这里暂不跑真实 MWPM 代码，仅记录 ML 模型的行为。
            
            print(f"Code Distance L={L_test}:")
            print(f"  > FNO Accuracy: {acc_f:.4f} (Error Rate: {1-acc_f:.4f})")
            print(f"  > CNN Accuracy: {acc_c:.4f} (Error Rate: {1-acc_c:.4f})")
            
            results['L'].append(L_test)
            results['FNO'].append(1-acc_f)
            results['CNN'].append(1-acc_c)

    # 5. 简单绘图
    plt.figure(figsize=(8, 6))
    plt.plot(results['L'], results['FNO'], 'o-', label='FNO (Trained on L=5)', linewidth=2)
    plt.plot(results['L'], results['CNN'], 's--', label='CNN (Trained on L=5)', linewidth=2)
    plt.xlabel('Code Distance (L)')
    plt.ylabel('Logical Error Rate')
    plt.title(f'Zero-Shot Generalization (GKP Noise sigma={SIGMA_TRAIN})')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()

    # Save figure to avoid QThreadStorage warnings on Windows
    output_file = 'generalization_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()  # Properly close the figure

if __name__ == '__main__':
    # 固定种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)
    run_experiment()