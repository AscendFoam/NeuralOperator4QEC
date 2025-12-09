## prompt:
ok，我现在想试试做一个快速验证FNO在GKP码上对比传统解码方式是否有优势的实验，给出相应的实验方案（不用给出具体代码，给出详细方案即可）

## answer:
这是一个非常务实的需求。为了“快速验证”且“直击痛点”，我们不需要一上来就搞极其复杂的全电路级噪声（Circuit-level Noise）或者多模态系统。

我们将选择**“GKP-Surface Code 在唯象噪声（Phenomenological Noise）下的解码”**作为切入点。这是目前最标准、数据生成最快、且最容易看出 FNO（傅里叶神经算子） 优势的场景。

以下是为你设计的**快速验证实验方案 (Proof-of-Concept Plan)**：

---

### 实验核心目标
1.  **证明有效性：** FNO 能否利用 GKP 的模拟信息（Analog Info）达到或超过传统 Analog MWPM 的解码阈值？
2.  **证明泛化性（关键差异点）：** 在 $d=5$ 上训练的模型，能否**不经微调**直接解 $d=7$ 或 $d=9$ 的码？（这是 CNN 做不到而 FNO 能做到的）。

---

### 第一步：数据生成 (Data Generation)

为了快速，我们使用**唯象噪声模型**（假设测量本身含噪，但不模拟具体的 CNOT 门错误）。

* **系统配置：**
    * **码型：** Rotated Surface Code（旋转表面码），每个数据量子比特是一个 GKP 态。
    * **码距：** $d \in \{5, 7, 9\}$。
* **噪声模拟 (Simulation)：**
    * **GKP 层噪声：** 对每个格点施加随机高斯位移 $n \sim \mathcal{N}(0, \sigma^2)$。$\sigma$ 对应 GKP 的压缩度（Squeezing, dB）。
    * **症候群提取：**
        * 计算每个校验子（Stabilizer）的理想模拟值：$S_{ideal} = \sum n_i$。
        * 加上测量噪声：$S_{meas} = S_{ideal} + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, \sigma_{meas}^2)$。
        * **关键处理：** GKP 的测量结果是模 $\sqrt{\pi}$ 的。所以最终症候群是 $S_{final} = (S_{meas} \pmod{\sqrt{\pi}})$。
* **数据集构建：**
    * **输入 $X$：** 形状为 $(N, d, d, 4)$。
        * 包含 X-stabilizers 和 Z-stabilizers 的模拟值。
        * **特征工程（至关重要）：** 不要直接输入数值！由于 GKP 是周期性的，必须做**周期性嵌入**。将每个点的测量值 $s$ 映射为 $[\sin(C s), \cos(C s)]$。因为有 X 和 Z 两种校验子，所以每个点有 4 个通道。
    * **标签 $Y$：** 形状为 $(N, 4)$ 或 $(N, 1)$。
        * 通过运行一个完美的参考解码器（如 MWPM）对比，计算实际发生的逻辑错误类别（I, X, Y, Z）。
    * **规模：** 训练集（$d=5$）生成 50,000 条；测试集（$d=5, 7, 9$）各生成 5,000 条。

---

### 第二步：模型构建 (Model Setup)

你需要实现三个模型进行“打擂台”。

#### 1. 挑战者：傅立叶神经算子 (FNO-2D)
* **架构：**
    * **Lifting Layer：** `Linear(4, 64)`，将周期特征映射到高维。
    * **Fourier Layers：** 堆叠 3-4 层。
        * 每层包含：2D FFT $\rightarrow$ 频域乘法（只保留低频 modes，例如前 12-16 个）$\rightarrow$ 2D IFFT $\rightarrow$ GELU 激活。
        * *注意：* 不需要 Padding，因为我们利用 FFT 也就是利用了隐式的周期性边界（对于 Toric code 是完美的，对于 Surface code 也可以近似）。
    * **Projection Layer：** `Linear(64, 4)` 输出逻辑错误概率。
* **配置：** 优化器 AdamW，学习率 1e-3，Weight Decay 1e-4。

#### 2. 深度学习基准：卷积神经网络 (CNN)
* **架构：** 标准的 ResNet-18 或 5 层简单的 CNN。
* **关键设置：** 必须使用 **Circular Padding (循环填充)** 而不是 Zero Padding，以公平对比 FNO 的周期性处理能力。
* **输入：** 同样的 $(d, d, 4)$ 周期性嵌入数据。

#### 3. 传统基准：模拟 MWPM (Analog MWPM)
* **工具：** 使用 `PyMatching` 库。
* **设置：**
    * 这不是机器学习模型，不需要训练。
    * **权重计算：** 你需要根据 GKP 的噪声方差 $\sigma^2$，手动计算每条边的权重 $w = \log(\frac{P(\text{no error})}{P(\text{error})})$。因为输入是模拟值，这里的概率 $P$ 是高斯概率密度函数的积分。
    * 这是目前的 SOTA“及格线”。

---

### 第三步：实验流程 (Execution Steps)

#### 实验 A：同分布性能对比 (In-distribution Performance)
* **操作：** 在 $d=5$ 数据集上训练 FNO 和 CNN。
* **测试：** 在 $d=5$ 的测试集上，扫描不同的噪声强度（例如 7dB 到 11dB）。
* **验证点：**
    1.  画出逻辑错误率 ($P_L$) 曲线。
    2.  FNO 是否比 CNN 收敛更快？
    3.  FNO 能否击败 Analog MWPM？（通常 MLP/CNN 只能持平，如果 FNO 能超越，就是巨大亮点）。

#### 实验 B：零样本超分辨率 (Zero-shot Super-Resolution) —— **核心验证**
* **操作：** 拿**刚刚在 $d=5$ 上训练好**的 FNO 和 CNN 模型。
* **测试：** **直接**在 $d=7$ 和 $d=9$ 的测试集上推理（Inference）。
* **预期结果（验证成功标志）：**
    * **CNN：** 可能会报错（维度不匹配）或者性能极差（因为它学的是局部的 $3\times3$ 卷积核，对全局拓扑变化不敏感）。
    * **FNO：** 应该能直接运行，并且给出合理的解码结果。如果 $d=9$ 的错误率比 $d=5$ 低（在过阈值区），说明它学到了“纠错的物理算子”而非“死记硬背”。
    * **MWPM：** 作为参考线，重新跑一遍 $d=7/9$ 的匹配。

---

### 第四步：评估与结果分析

验证实验成功的标志是获得以下图表：

1.  **阈值图 (Threshold Plot)：**
    * X轴：GKP Squeezing (dB)
    * Y轴：Logical Error Rate
    * 包含曲线：`MWPM (d=5)`, `CNN (d=5)`, `FNO (d=5)`。
    * *成功标志：* FNO 曲线在 MWPM 下方。

2.  **泛化能力图 (Generalization Plot)：**
    * X轴：Code Distance ($d$)
    * Y轴：Logical Error Rate (固定噪声，例如 9dB)
    * 包含曲线：`MWPM` (理论值), `FNO (Train d=5, Test d=all)`, `CNN (Train d=5, Test d=all)`。
    * *成功标志：* FNO 曲线随 $d$ 增加而下降（即有效抑制错误），而 CNN 曲线持平或上升（泛化失败）。

### 总结建议

* **不要碰** Circuit-level noise（暂时）。那个模拟太慢，且调试复杂。
* **务必做** 周期性嵌入（sin/cos input）。这是处理 GKP 数据的诀窍。
* **重点看** 泛化实验。这是审稿人最想看到的“神经算子相比于 CNN 的独特优势”。

这个方案大约 1-2 周的业余时间即可跑通（如果熟悉 Python/PyTorch），能最快速度告诉你 FNO 这条路通不通。