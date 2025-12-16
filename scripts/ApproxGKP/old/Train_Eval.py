from FNO import FNO2dDecoder
from GKP_Simulator import GKPPhysicsSimulator
import torch
import torch.nn as nn
import numpy as np



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