import numpy as np
import qutip as qt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. 物理层：高精度 GKP 模拟器
# ==========================================
class HighResGKPSimulator:
    def __init__(self, N=80, delta=0.3):
        self.N = N
        self.delta = delta
        self.sqrt_pi = np.sqrt(np.pi)
        # 预计算逻辑态（简化版，仅用于生成Wigner背景）
        self.xvec = np.linspace(-7, 7, 64) # 提高分辨率到 64
        self.logical_0, self.logical_1 = self._precompute_states()

    def _precompute_states(self):
        # 构造理想叠加的高斯波包近似
        # 为了速度，直接在相空间构造 Wigner 函数的近似形式是可行的，
        # 但为了严谨，我们还是用 QuTiP 构造态矢量
        states = []
        for logic in [0, 1]:
            psi = qt.Qobj(np.zeros(self.N), dims=[[self.N], [1]])
            n_max = int(np.sqrt(self.N)/(2*self.sqrt_pi)) + 2
            z = -np.log(self.delta)
            for k in range(-n_max, n_max + 1):
                shift = (2 * k + logic) * self.sqrt_pi
                weight = np.exp(-0.5 * (self.delta * shift)**2)
                alpha = shift / np.sqrt(2)
                psi += weight * qt.displace(self.N, alpha) * qt.squeeze(self.N, z) * qt.basis(self.N, 0)
            states.append(psi.unit())
        return states[0], states[1]

    def generate_dataset(self, n_samples, noise_sigma, batch_mode=False):
        """
        生成数据集。
        为了对比，我们需要知道真实的噪声位移 (u, v)。
        """
        wigners = []
        shifts = [] # 真实施加的噪声
        
        # 为了进度条显示
        iterator = range(n_samples) if batch_mode else tqdm(range(n_samples), desc="Generating Data")
        
        for _ in iterator:
            # 1. 随机态
            if np.random.rand() > 0.5:
                psi = self.logical_0
            else:
                psi = self.logical_1
            
            # 2. 施加随机高斯位移噪声 (u, v)
            u = np.random.normal(0, noise_sigma)
            v = np.random.normal(0, noise_sigma)
            
            # 物理上施加噪声
            zeta = (u + 1j * v) / np.sqrt(2)
            noisy_psi = qt.displace(self.N, zeta) * psi
            
            # 3. 计算 Wigner (这是最慢的一步)
            W = qt.wigner(noisy_psi, self.xvec, self.xvec)
            wigners.append(W)
            shifts.append([u, v])
            
        return torch.tensor(np.array(wigners), dtype=torch.float32).unsqueeze(1), \
               torch.tensor(np.array(shifts), dtype=torch.float32)

# ==========================================
# 2. 模型层：增强版 FNO
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2dCorrector(nn.Module):
    def __init__(self, modes=16, width=64): # 增加 modes 和 width
        super(FNO2dCorrector, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        self.p = nn.Linear(1, self.width) 
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2) 

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# ==========================================
# 3. 评估层：Baseline 对比与逻辑错误率
# ==========================================
def calculate_logical_error_rate(predicted_shifts, true_shifts):
    """
    计算逻辑错误率。
    对于 GKP，如果残余位移 |u_true - u_pred| > sqrt(pi)/2，则发生逻辑错误。
    """
    sqrt_pi = np.sqrt(np.pi)
    threshold = sqrt_pi / 2
    
    # 残余位移
    residual = true_shifts - predicted_shifts
    
    # 考虑 GKP 的周期性：我们需要把残余位移 map 到 [-sqrt(pi)/2, sqrt(pi)/2]
    # 但在这里，我们要判断的是纠错是否"失败"。
    # 简单的判据：纠错后的点是否比不纠错更远离原点？或者是否跨越了 Voronoi 边界。
    # 标准判据：check if round((u_true - u_pred)/sqrt(pi)) != 0
    
    # 对 q (u) 和 p (v) 分别判断
    errors_u = np.abs(residual[:, 0]) > threshold
    errors_v = np.abs(residual[:, 1]) > threshold
    
    # 任何一个方向出错都算逻辑错误
    total_errors = np.logical_or(errors_u, errors_v)
    return np.mean(total_errors.numpy())

def baseline_homodyne(true_shifts, sigma):
    """
    Baseline: Homodyne Detection + Binning
    模拟：测量值 m = u_true + noise_meas
    这里的 noise_meas 取决于 homodyne 效率，这里为了公平对比，假设 homodyne 完美，
    只受限于 GKP 本身的有限挤压带来的模糊。
    实际上，对于位移噪声模型，Homodyne 策略就是把测量结果 round 到最近的 sqrt(pi)。
    """
    # 模拟测量值：真实位移 + 测量噪声(这里假设测量本身完美，只纠正位移)
    # 传统的解码：predicted_shift = m mod sqrt(pi) -> centered
    # 等价于：predicted_shift = m - round(m / sqrt(pi)) * sqrt(pi)
    # 但我们这里是已知 true_shifts (即漂移量)，我们想看如果不纠正 (Identity) 或者用简单策略纠正的效果
    
    # 实际上，Homodyne 纠错是指：测量 q，得到 outcome，然后把态移回最近的格点。
    # 如果位移 u < sqrt(pi)/2，Homodyne 能完美测出 u 并移回去。
    # 如果 u > sqrt(pi)/2，Homodyne 会把它移到错误的格点。
    # 所以 Homodyne 的预测值其实就是 true_shifts (假设它能完美看到位移)，
    # 但受限于 threshold。
    
    # 我们用 "Identity" 作为 Baseline 1 (不做纠错)
    # 用 "Threshold Check" 作为 Baseline 2 (理论极限)
    pass

def run_benchmark():
    # 参数设置
    noise_levels = [0.15, 0.20, 0.25, 0.30] # sigma
    delta = 0.3
    n_train = 400
    n_test = 100
    
    sim = HighResGKPSimulator(delta=delta)
    
    # 结果存储
    results_fno = []
    results_uncorrected = []
    
    print(f"--- Starting Benchmark with Delta={delta} ---")
    
    for sigma in noise_levels:
        print(f"\nEvaluating Noise Sigma = {sigma}")
        
        # 1. 生成数据
        print("Generating training data...")
        train_x, train_y = sim.generate_dataset(n_train, sigma, batch_mode=True)
        print("Generating test data...")
        test_x, test_y = sim.generate_dataset(n_test, sigma, batch_mode=True)
        
        # 2. 训练 FNO
        model = FNO2dCorrector(modes=12, width=32) # 显存不足可调小
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        criterion = nn.MSELoss()
        
        # 快速训练 loop
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            out = model(train_x)
            loss = criterion(out, train_y)
            loss.backward()
            optimizer.step()
        
        # 3. 测试与对比
        model.eval()
        with torch.no_grad():
            pred_fno = model(test_x)
            
            # FNO 逻辑错误率
            ler_fno = calculate_logical_error_rate(pred_fno, test_y)
            
            # Baseline: Uncorrected (不做任何操作，看噪声是否直接让态跑偏)
            # 预测位移为 0
            pred_baseline = torch.zeros_like(test_y)
            ler_baseline = calculate_logical_error_rate(pred_baseline, test_y)
            
            results_fno.append(ler_fno)
            results_uncorrected.append(ler_baseline)
            
            print(f"-> Uncorrected LER: {ler_baseline:.4f}")
            print(f"-> FNO LER        : {ler_fno:.4f}")

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, results_uncorrected, 'k--o', label='No Correction (Baseline)')
    plt.plot(noise_levels, results_fno, 'r-^', label='FNO Correction')
    
    # 添加理论辅助线 (Small displacement approx)
    # P_err ~ 1/2 * erfc(sqrt(pi)/(2*sqrt(2)*sigma))
    from scipy.special import erfc
    theory_err = [0.5 * erfc(np.sqrt(np.pi)/(2*np.sqrt(2)*s)) * 2 for s in noise_levels] # *2 for both u and v
    plt.plot(noise_levels, theory_err, 'b:', label='Theoretical Limit (Displacement)')
    
    plt.xlabel('Noise Standard Deviation ($\sigma$)')
    plt.ylabel('Logical Error Rate (LER)')
    plt.title(f'GKP Error Correction Performance (Delta={delta})')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_benchmark()