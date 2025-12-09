这是一个非常扎实的实验计划。考虑到你的背景是人工智能与量子计算的交叉领域，这份代码实现将分为三个核心模块：

1.  **物理仿真层 (Physics Layer)**：基于 `QuTiP` 实现近似 GKP 态的制备与噪声注入，严格遵循论文 Sec II B 的定义。
2.  **神经算子层 (Model Layer)**：构建基于 PyTorch 的 Fourier Neural Operator (FNO) 架构，专门用于处理 Wigner 函数图像并回归出纠错位移。
3.  **训练与验证 (Execution)**：连接物理数据与神经网络的训练闭环。

### 环境依赖

```bash
pip install qutip torch numpy matplotlib scipy
```

-----

### 1\. 物理仿真层：近似 GKP 态生成器

这段代码实现了论文 Eq. (11) [cite_start][cite: 158] 描述的物理 GKP 态（Comb of Squeezed States with Envelope）。

```python
import numpy as np
import qutip as qt
import torch

class GKPPhysicsSimulator:
    def __init__(self, N=60, delta=0.3):
        """
        初始化 GKP 物理模拟器。
        Args:
            N (int): Hilbert空间截断维数。
            [cite_start]delta (float): 近似参数 Delta (squeezing parameter) [cite: 148, 164]。
                           Delta 越小，态越接近理想 GKP 态，但物理制备越难。
        """
        self.N = N
        self.delta = delta
        self.sqrt_pi = np.sqrt(np.pi)
        
        # 预先构建逻辑态 |0_L> 和 |1_L> 的近似形式
        # [cite_start]使用 Eq. (11): Sum of squeezed states with Gaussian envelope [cite: 158]
        self.logical_0 = self._construct_approx_state(logical_val=0)
        self.logical_1 = self._construct_approx_state(logical_val=1)

    def _construct_approx_state(self, logical_val):
        """
        构建近似 GKP 码字。
        实现逻辑：叠加一系列位移后的挤压真空态，并施加高斯包络权重。
        """
        psi = qt.Qobj(np.zeros(self.N), dims=[[self.N], [1]])
        
        # 晶格常数通常为 2*sqrt(pi)，这里求和范围取决于截断维数 N
        # 只有在 N 范围内有显著光子数的项才会被保留
        n_max = int(np.sqrt(self.N) / (2 * self.sqrt_pi)) + 2
        
        # 定义挤压态的挤压因子。理想 GKP 是高度挤压态。
        # 这里简化处理：近似态由一系列有限挤压的高斯波包组成
        squeezing_db = -10 * np.log10(self.delta**2) # 将 Delta 转换为近似 dB
        z = -np.log(self.delta) # 粗略估计挤压参数 r
        
        for k in range(-n_max, n_max + 1):
            # [cite_start]晶格点位置: (2k + mu) * sqrt(pi) [cite: 96]
            shift = (2 * k + logical_val) * self.sqrt_pi
            
            # 权重: Gaussian Envelope Eq. (11) [cite_start][cite: 158]
            weight = np.exp(-0.5 * (self.delta * shift)**2)
            
            # 位移算符 D(alpha) 作用在挤压真空上
            # 注意：QuTiP 中 displace(N, alpha)，这里 alpha 是相空间位移
            # 对于位置本征态位移 x，对应的 alpha = x / sqrt(2) (实部)
            alpha = shift / np.sqrt(2)
            
            # 生成挤压真空并位移
            state = qt.displace(self.N, alpha) * qt.squeeze(self.N, z) * qt.basis(self.N, 0)
            psi += weight * state
            
        return psi.unit() # 归一化

    def generate_batch(self, batch_size, noise_sigma=0.1, grid_size=32):
        """
        生成一批用于训练的数据。
        Args:
            batch_size: 批大小
            [cite_start]noise_sigma: 位移噪声的标准差 sigma [cite: 673]
            grid_size: Wigner 函数的网格大小
        Returns:
            wigners_tensor: (Batch, 1, Grid, Grid)
            displacements_tensor: (Batch, 2) - 真实的位移向量 (u, v)
        """
        wigners = []
        labels = []
        
        xvec = np.linspace(-6, 6, grid_size)
        
        for _ in range(batch_size):
            # 1. 随机选择逻辑态或叠加态
            coeff = np.random.rand()
            if np.random.rand() > 0.5:
                # 随机叠加态 alpha|0> + beta|1>
                theta = np.random.uniform(0, 2*np.pi)
                psi = np.cos(theta)*self.logical_0 + np.sin(theta)*self.logical_1
            else:
                psi = self.logical_0 if np.random.rand() > 0.5 else self.logical_1
            
            # [cite_start]2. 注入高斯位移噪声 Eq. (26) [cite: 673]
            # 噪声形式 D(zeta)，zeta = (u + iv)/sqrt(2)
            # u, v ~ N(0, sigma^2)
            u = np.random.normal(0, noise_sigma)
            v = np.random.normal(0, noise_sigma)
            zeta = (u + 1j * v) / np.sqrt(2)
            
            # 应用噪声 D(zeta) |psi>
            noisy_psi = qt.displace(self.N, zeta) * psi
            
            # 3. 计算 Wigner 函数作为 FNO 输入
            # 注意：Wigner 计算较慢，实际大规模训练建议预生成存为 .npy
            W = qt.wigner(noisy_psi, xvec, xvec)
            
            wigners.append(W)
            # 标签是我们需要"消除"的位移，即 -u, -v，或者是刚才施加的 u, v
            # 这里我们让网络预测施加的噪声 (u, v)，解码时反向操作即可
            labels.append([u, v])
            
        # 转换为 PyTorch Tensor
        wigners_tensor = torch.tensor(np.array(wigners), dtype=torch.float32).unsqueeze(1) # Add channel dim
        labels_tensor = torch.tensor(np.array(labels), dtype=torch.float32)
        
        return wigners_tensor, labels_tensor

# 测试一下物理层
# sim = GKPPhysicsSimulator(N=40, delta=0.3)
# w, l = sim.generate_batch(2)
# print(f"Wigner shape: {w.shape}, Label: {l}")
```

-----

### 2\. 神经算子层：Spectral Decoder

这里我们实现一个改进的 2D FNO。标准的 FNO 输出也是一个场（Field），但在 GKP 纠错场景下（Syndrome Decoding），我们需要将全图信息坍缩为一个 2D 向量 $(\delta q, \delta p)$。

```python
import torch.nn as nn
import torch.nn.functional as F

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
```

-----

### 3\. 训练闭环

[cite_start]这里我们定义损失函数。注意，对于 GKP 码，位移是模 $\sqrt{\pi}$ 周期性的。如果噪声很小（小于 $\sqrt{\pi}/2$），直接使用 MSE 即可。如果噪声较大，需要设计 Modular MSE Loss。为简单起见，且根据论文 Sec IV 中提及的 "small displacement assumption" [cite: 170, 732]，我们在此使用 MSE。

```python
def train_fno_decoder():
    # 1. 配置参数
    N_HILBERT = 40
    DELTA = 0.3 # 约 10dB Squeezing，符合实验现状
    NOISE_SIGMA = 0.15 # 噪声强度
    BATCH_SIZE = 16 # 小批量演示
    EPOCHS = 100
    LR = 1e-3
    GRID_SIZE = 32 # 降低分辨率以加快演示速度

    # 2. 实例化组件
    print("Initializing Physics Simulator (this may take a few seconds)...")
    simulator = GKPPhysicsSimulator(N=N_HILBERT, delta=DELTA)
    
    model = FNO2dDecoder(modes=8, width=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("Start Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        # 实时生成数据 (Online Learning)
        # 这种方式能防止过拟合，因为网络永远不会看到完全相同的噪声样本
        inputs, targets = simulator.generate_batch(BATCH_SIZE, NOISE_SIGMA, GRID_SIZE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 计算 Loss: 预测的位移 vs 真实的位移噪声
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

    print("Training Finished.")
    
    # 简单的验证
    model.eval()
    with torch.no_grad():
        test_in, test_target = simulator.generate_batch(1, NOISE_SIGMA, GRID_SIZE)
        pred = model(test_in)
        print(f"\n--- Evaluation ---")
        print(f"True Noise (u, v): {test_target[0].numpy()}")
        print(f"FNO Predicted    : {pred[0].numpy()}")
        print(f"Error            : {np.abs(test_target[0].numpy() - pred[0].numpy())}")

# 运行训练
if __name__ == "__main__":
    train_fno_decoder()
```

### 代码关键点解析 (Research Insights)

1.  **物理嵌入 (Physical Embedding)**:

      * 在 `GKPPhysicsSimulator` 中，我没有简单地使用 `displace` 作用于真空态，而是显式地构建了 *Squeezed State Comb*。这是为了模拟论文 Sec II B 中提到的近似态的高斯包络效应 (Eq. [cite_start]8, 11) [cite: 138, 158]。这使得数据更符合真实实验场景。

2.  **FNO 的高频模态 (High Frequency Modes)**:

      * 在 `FNO2dDecoder` 中，`modes` 参数的选择至关重要。GKP 的 Wigner 函数具有尖锐的峰（Fig. [cite_start]1c [cite: 98]）。如果 FNO 的 `modes` 设置得太低（过度低通滤波），网络可能会抹平这些量子干涉峰，导致无法识别逻辑态。建议在显存允许的情况下尽可能保留较高频率的模态。

3.  **为什么是回归 (Regression) 而不是分类**:

      * [cite_start]尽管 GKP 是为了纠正到逻辑 0 或 1，但在 continuous-variable (CV) 体系中，纠错的第一步通常是 *Analog Information Decoding*，即估计连续的漂移量 $(\delta q, \delta p)$ 并移回去，然后再进行离散的 Pauli 测量。这种方法利用了所谓的 "Soft Information"，已被证明优于硬判决 [cite: 669, 670]。

### 下一步建议

你可以直接运行这段代码。如果要在论文中使用，建议将 `generate_batch` 中的 `wigner` 计算并行化（使用 `multiprocessing`），因为 QuTiP 的 Wigner 变换是 CPU 密集的，在大 Batch 下会成为瓶颈。