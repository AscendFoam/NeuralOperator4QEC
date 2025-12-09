import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

# ==========================================
# 1. 物理层: 带关联噪声的 GKP Toric Code
# ==========================================
class GKPToricCodeSim:
    def __init__(self, L):
        self.L = L
        self.sqrt_pi = np.sqrt(np.pi)
        
    def generate_batch(self, batch_size, sigma_noise):
        # 1. 生成基础白噪声
        raw_noise = np.random.normal(0, sigma_noise, (batch_size, 2, self.L, self.L))
        
        # 2. 引入关联 (Correlated Noise)
        # 关键点：关联长度 sigma_blur 设为 1.5
        # 这意味着错误团块大小约为 3-4 个格点，正好稍微超过 CNN 3x3 的感受野
        # 但对于 FNO (全局视野) 来说，这一览无余
        sigma_blur = 1.5 
        correlated_noise = np.zeros_like(raw_noise)
        
        # 使用 wrap 模式保持 Toric Code 的拓扑特性
        for i in range(batch_size):
            correlated_noise[i, 0] = scipy.ndimage.gaussian_filter(raw_noise[i, 0], sigma=sigma_blur, mode='wrap')
            correlated_noise[i, 1] = scipy.ndimage.gaussian_filter(raw_noise[i, 1], sigma=sigma_blur, mode='wrap')
            
        # 重新归一化能量，保证公平对比
        # 如果不归一化，平滑后的噪声会变很小，导致任务过于简单
        current_std = np.std(correlated_noise)
        noise = correlated_noise / (current_std + 1e-8) * sigma_noise
        
        # 3. 计算症候群 (Syndrome)
        s_z = (noise[:, 0, :, :] + 
               np.roll(noise[:, 0, :, :], -1, axis=1) + 
               noise[:, 1, :, :] + 
               np.roll(noise[:, 1, :, :], -1, axis=2))
        s_z_analog = (s_z + self.sqrt_pi/2) % self.sqrt_pi - self.sqrt_pi/2
        
        # 4. 计算逻辑错误 (Label)
        logical_err_1 = np.sum(noise[:, 0, :, 0], axis=1)
        y_1 = np.round(logical_err_1 / self.sqrt_pi).astype(int) % 2
        
        # 5. 特征工程
        scale = 2 * np.pi / self.sqrt_pi
        X = np.stack([np.sin(s_z_analog * scale), np.cos(s_z_analog * scale)], axis=1)
        
        return torch.FloatTensor(X), torch.FloatTensor(y_1).unsqueeze(1)

# ==========================================
# 2. FNO: 纯净的低频学习机
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes # 只需要一个 modes 参数，处理正方形

        scale = (1 / (in_channels * out_channels))
        # 权重只定义在最低频 modes x modes 上
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # 1. FFT
        x_ft = torch.fft.rfft2(x)
        
        # 2. 频域乘法 (分辨率无关的核心)
        # 不管输入 x 是 5x5 还是 9x9，x_ft 的低频部分我们只取前 modes 个
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # 动态截断：确保 modes 不超过当前输入的尺寸
        current_modes_x = min(self.modes, x_ft.size(-2))
        current_modes_y = min(self.modes, x_ft.size(-1))
        
        out_ft[:, :, :current_modes_x, :current_modes_y] = \
            self.compl_mul2d(x_ft[:, :, :current_modes_x, :current_modes_y], 
                             self.weights1[:, :, :current_modes_x, :current_modes_y])
                             
        out_ft[:, :, -current_modes_x:, :current_modes_y] = \
            self.compl_mul2d(x_ft[:, :, -current_modes_x:, :current_modes_y], 
                             self.weights2[:, :, :current_modes_x, :current_modes_y])

        # 3. IFFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d_Pure(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        self.modes = modes
        self.width = width
        self.p = nn.Conv2d(2, self.width, 1) # Lifting
        
        # 4层 FNO
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.q1 = nn.Linear(self.width, 128)
        self.q2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.p(x)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        
        # 全局平均池化 (适应任意尺寸)
        x = x.mean(dim=(2, 3))
        x = F.gelu(self.q1(x))
        x = self.q2(x)
        return torch.sigmoid(x)

# ==========================================
# 3. CNN Baseline (标准结构)
# ==========================================
class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # 3层 CNN，感受野 3+2+2 = 7 (但在 3x3 Torus padding 下行为复杂)
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1, padding_mode='circular')
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, padding_mode='circular')
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='circular')
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=(2,3))
        return torch.sigmoid(self.fc(x))

# ==========================================
# 4. 主程序
# ==========================================
def run_final_check():
    # 配置
    TRAIN_L = 5
    TEST_L_LIST = [5, 7, 9, 11, 13]
    TRAIN_SIZE = 10000 
    TEST_SIZE = 2000
    BATCH_SIZE = 128
    EPOCHS = 15
    
    # 关键参数：低噪声 + 强关联
    SIGMA_TRAIN = 0.15 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}. Sigma={SIGMA_TRAIN} (Correlated Noise)")

    # 1. 数据
    sim_train = GKPToricCodeSim(TRAIN_L)
    x_train, y_train = sim_train.generate_batch(TRAIN_SIZE, SIGMA_TRAIN)
    train_loader = DataLoader(TensorDataset(x_train.to(device), y_train.to(device)), batch_size=BATCH_SIZE, shuffle=True)

    # 2. 模型
    # Modes=2: 强迫模型只学最宏观的拓扑性质，忽略高频细节
    fno = FNO2d_Pure(modes=2, width=64).to(device)
    cnn = CNNBaseline().to(device)
    
    opt_fno = optim.AdamW(fno.parameters(), lr=0.002)
    opt_cnn = optim.AdamW(cnn.parameters(), lr=0.002)
    criterion = nn.BCELoss()

    # 3. 训练
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        fno.train(); cnn.train()
        loss_f = 0; loss_c = 0
        
        for bx, by in train_loader:
            # FNO
            opt_fno.zero_grad()
            l_f = criterion(fno(bx), by)
            l_f.backward()
            opt_fno.step()
            loss_f += l_f.item()
            
            # CNN
            opt_cnn.zero_grad()
            l_c = criterion(cnn(bx), by)
            l_c.backward()
            opt_cnn.step()
            loss_c += l_c.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | FNO Loss: {loss_f/len(train_loader):.4f} | CNN Loss: {loss_c/len(train_loader):.4f}")

    # 4. 测试
    print("\nStarting Evaluation...")
    fno.eval(); cnn.eval()
    
    with torch.no_grad():
        for L in TEST_L_LIST:
            sim = GKPToricCodeSim(L)
            xt, yt = sim.generate_batch(TEST_SIZE, SIGMA_TRAIN)
            xt, yt = xt.to(device), yt.to(device)
            
            acc_f = ((fno(xt) > 0.5) == yt).float().mean().item()
            acc_c = ((cnn(xt) > 0.5) == yt).float().mean().item()
            
            print(f"L={L} | FNO Acc: {acc_f:.4f} | CNN Acc: {acc_c:.4f}")

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    run_final_check()