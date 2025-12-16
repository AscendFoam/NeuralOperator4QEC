import numpy as np
import qutip as qt
import torch

class GKPPhysicsSimulator:
    def __init__(self, N=60, delta=0.3):
        """
        初始化 GKP 物理模拟器。
        Args:
            N (int): Hilbert空间截断维数。
            delta (float): 近似参数 Delta (squeezing parameter) [cite: 148, 164]。
                           Delta 越小，态越接近理想 GKP 态，但物理制备越难。
        """
        self.N = N
        self.delta = delta
        self.sqrt_pi = np.sqrt(np.pi)
        
        # 预先构建逻辑态 |0_L> 和 |1_L> 的近似形式
        # 使用 Eq. (11): Sum of squeezed states with Gaussian envelope 
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
            # 晶格点位置: (2k + mu) * sqrt(pi) [cite: 96]
            shift = (2 * k + logical_val) * self.sqrt_pi
            
            # 权重: Gaussian Envelope Eq. (11) 
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
            noise_sigma: 位移噪声的标准差 sigma [cite: 673]
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
            
            # 2. 注入高斯位移噪声 Eq. (26) [cite: 673]
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