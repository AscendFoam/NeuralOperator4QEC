import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 物理模拟器 (保持不变)
# ==========================================
import scipy.ndimage # 需要引入这个库

class GKPToricCodeSim:
    def __init__(self, L):
        self.L = L
        self.sqrt_pi = np.sqrt(np.pi)
        
    def generate_batch(self, batch_size, sigma_noise):
        # --- 修改开始：引入关联噪声 (Correlated Noise) ---
        # 1. 先生成白噪声
        raw_noise = np.random.normal(0, sigma_noise, (batch_size, 2, self.L, self.L))
        
        # 2. 施加高斯模糊，产生空间关联
        # correlation_scale 控制关联的范围。
        # 关键：为了验证泛化性，关联范围应相对于 L 保持一定比例，或者固定物理尺寸。
        # 这里我们让关联长度约为 L 的 1/3，这样在 L=9 时模型必须理解全局云团
        sigma_blur = 0.25 * self.L 
        
        correlated_noise = np.zeros_like(raw_noise)
        for i in range(batch_size):
            # 对每个样本的每个通道做平滑
            # mode='wrap' 对应环面码的周期性边界
            correlated_noise[i, 0] = scipy.ndimage.gaussian_filter(raw_noise[i, 0], sigma=sigma_blur, mode='wrap')
            correlated_noise[i, 1] = scipy.ndimage.gaussian_filter(raw_noise[i, 1], sigma=sigma_blur, mode='wrap')
            
        # 3. 重新归一化 (Renormalize)
        # 平滑会降低方差，必须乘回来，否则噪声太小了解码太容易
        current_std = np.std(correlated_noise)
        correlated_noise = correlated_noise / (current_std + 1e-8) * sigma_noise
        
        noise = correlated_noise
        # --- 修改结束 ---
        
        # 下面保持不变，计算 Syndrome 和 Logic Error
        s_z = (noise[:, 0, :, :] + 
               np.roll(noise[:, 0, :, :], -1, axis=1) + 
               noise[:, 1, :, :] + 
               np.roll(noise[:, 1, :, :], -1, axis=2))
        s_z_analog = (s_z + self.sqrt_pi/2) % self.sqrt_pi - self.sqrt_pi/2
        
        logical_err_1 = np.sum(noise[:, 0, :, 0], axis=1)
        y_1 = np.round(logical_err_1 / self.sqrt_pi).astype(int) % 2
        
        scale = 2 * np.pi / self.sqrt_pi
        feat_sin = np.sin(s_z_analog * scale)
        feat_cos = np.cos(s_z_analog * scale)
        X = np.stack([feat_sin, feat_cos], axis=1) 
        
        return torch.FloatTensor(X), torch.FloatTensor(y_1).unsqueeze(1)
# ==========================================
# 2. CNN Baseline (保持不变，处理原始网格)
# ==========================================
class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # 简单的 3层 CNN，专门针对 Grid
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1, padding_mode='circular')
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, padding_mode='circular')
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='circular')
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=(2,3)) # Global Pooling
        return torch.sigmoid(self.fc(x))

# ==========================================
# 3. FNO (标准版，但输入固定为 32x32)
# ==========================================
class SpectralConv2d(nn.Module):
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
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d_Continuum(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        
        self.p = nn.Conv2d(2, self.width, 1) 
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.q1 = nn.Linear(self.width, 128)
        self.q2 = nn.Linear(128, 1)

    def forward(self, x):
        # x input is strictly 32x32
        x = self.p(x)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        
        x = x.mean(dim=(2, 3))
        x = F.gelu(self.q1(x))
        x = self.q2(x)
        return torch.sigmoid(x)

# ==========================================
# 4. 实验主逻辑 (关键修改区)
# ==========================================
def run_experiment():
    # 配置
    TRAIN_L = 5
    TEST_L_LIST = [5, 7, 9] 
    FIXED_SIZE = 32 # [关键] FNO 的统一“视网膜”大小
    TRAIN_SIZE = 20000 # 增加数据量保证收敛
    TEST_SIZE = 2000
    EPOCHS = 20
    BATCH_SIZE = 128
    SIGMA_TRAIN = 0.20 # ~8.7dB
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}. FNO Input Resolution Fixed to {FIXED_SIZE}x{FIXED_SIZE}")

    # 1. 数据生成
    print(f"Generating Training Data (L={TRAIN_L}, sigma={SIGMA_TRAIN})...")
    sim_train = GKPToricCodeSim(TRAIN_L)
    x_train_raw, y_train = sim_train.generate_batch(TRAIN_SIZE, SIGMA_TRAIN)
    
    # 2. 模型初始化
    # FNO 能够处理 32x32 的输入，modes 设置为 8 (覆盖低频到中频)
    fno = FNO2d_Continuum(modes=10, width=64).to(device)
    cnn = CNNBaseline().to(device)
    
    opt_fno = optim.AdamW(fno.parameters(), lr=0.001, weight_decay=1e-4)
    opt_cnn = optim.AdamW(cnn.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()

    train_ds = TensorDataset(x_train_raw.to(device), y_train.to(device))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 训练
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        fno.train()
        cnn.train()
        loss_f_acc = 0.0
        loss_c_acc = 0.0
        
        for bx, by in train_loader:
            # --- CNN 训练 (原汁原味 L=5) ---
            opt_cnn.zero_grad()
            pred_c = cnn(bx)
            l_c = criterion(pred_c, by)
            l_c.backward()
            opt_cnn.step()
            loss_c_acc += l_c.item()

            # --- FNO 训练 (插值到 32x32) ---
            # 使用 bicubic 插值让原本离散的 Grid 变成连续场
            bx_resized = F.interpolate(bx, size=(FIXED_SIZE, FIXED_SIZE), mode='bilinear', align_corners=True)
            
            opt_fno.zero_grad()
            pred_f = fno(bx_resized)
            l_f = criterion(pred_f, by)
            l_f.backward()
            opt_fno.step()
            loss_f_acc += l_f.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | FNO Loss: {loss_f_acc/len(train_loader):.4f} | CNN Loss: {loss_c_acc/len(train_loader):.4f}")

    # 4. 泛化测试
    print("\nStarting Evaluation (Zero-Shot Generalization)...")
    
    fno.eval()
    cnn.eval()
    
    with torch.no_grad():
        for L_test in TEST_L_LIST:
            sim_test = GKPToricCodeSim(L_test)
            x_test, y_test = sim_test.generate_batch(TEST_SIZE, SIGMA_TRAIN)
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            # CNN: 直接跑在 L_test 上 (它会尝试用 padding 补救，但拓扑改变了)
            pred_c = cnn(x_test)
            acc_c = ((pred_c > 0.5) == y_test).float().mean().item()
            
            # FNO: 统统缩放到 32x32
            # 这里的魔法是：L=9 的逻辑错误链，被缩放到 32x32 后，
            # 和 L=5 的逻辑错误链在“视觉”上是一样的（都是贯穿全图的一条线）
            x_test_resized = F.interpolate(x_test, size=(FIXED_SIZE, FIXED_SIZE), mode='bilinear', align_corners=True)
            pred_f = fno(x_test_resized)
            acc_f = ((pred_f > 0.5) == y_test).float().mean().item()
            
            print(f"Code Distance L={L_test}:")
            print(f"  > FNO (Interp) Accuracy: {acc_f:.4f} (Error Rate: {1-acc_f:.4f})")
            print(f"  > CNN (Raw)    Accuracy: {acc_c:.4f} (Error Rate: {1-acc_c:.4f})")

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    run_experiment()