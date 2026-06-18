# 肿瘤图像分割

基于 **U-Net** 的肿瘤图像二值分割项目，使用 PyTorch 实现，包含数据加载、模型定义、训练与预测可视化全流程。

## 项目结构

```
肿瘤图像分割/
└── main.py          # 完整流程脚本（数据加载 / 模型 / 训练 / 预测）
```

## 环境依赖

- Python 3.8+
- PyTorch
- torchvision
- torchkeras
- torchsummary
- matplotlib
- Pillow

安装示例：

```bash
pip install torch torchvision torchkeras torchsummary matplotlib pillow
```

## 数据集结构

脚本默认从以下路径加载数据（Windows 风格绝对路径，路径可在脚本中按需修改）：

```
tumor_data/
├── train/
│   ├── patient/    # 训练原图（*）
│   └── tumor/      # 训练标签/掩码（*）
└── test/
    ├── patient/    # 测试原图（*）
    └── tumor/      # 测试标签/掩码（*）
```

图像与掩码一一对应，文件名按顺序匹配。

## 核心模块

### 1. `LoadDataset`
继承自 `torch.utils.data.Dataset`，使用 `glob` 获取图像与掩码路径列表，读入灰度图（`convert('L')`），分别通过 `transform` 与 `target_transform` 预处理：

- 将图像缩放至 `256 × 256`
- 转为 Tensor

### 2. DataLoader
- `batch_size = 8`
- 训练集打乱，测试集打乱

### 3. U-Net 模型
自定义实现，由 4 次下采样（`Conv` 块 + `MaxPool2d`）和 4 次上采样（`ConvTranspose2d` + 跳跃连接）组成：

- 通道数：1 → 64 → 128 → 256 → 512 → 1024
- 每个 `Conv` 块包含两次 `Conv2d(3×3) + BatchNorm2d + ReLU`
- 跳跃连接通过 `torch.cat` 在通道维拼接
- 输出层为 `Conv2d(64, 1, 3, 1, 1)`

### 4. 训练
使用 `torchkeras.KerasModel` 封装训练流程：

- 损失函数：`BCEWithLogitsLoss`
- 优化器：`Adam (lr=0.005)`
- 早停：`patience=10`，监控 `val_loss`
- 训练轮次：`epochs=10`
- 检查点保存至 `ckpt_path`

### 5. 模型导入与预测
- 重新实例化 `Unet` 与 `KerasModel`
- 通过 `model.load_state_dict(...)` 加载权重
- 遍历测试集，使用 `Sigmoid` 还原概率图，与原图、真实掩码一并展示

## 使用方式

由于脚本以 Jupyter Cell 风格编写（带 `# %%` 分隔），推荐在 **VS Code** 或 **Jupyter Notebook / Lab** 中打开 `main.py` 逐块运行：

1. 修改脚本顶部的数据路径与 `ckpt_path`，使其指向本机目录。
2. 按顺序执行各 Cell：数据加载 → 模型构建 → 模型摘要 → 训练 → 加载权重 → 预测可视化。
3. 如需保存最终模型，可取消注释：

   ```python
   # torch.save(model.state_dict(), 'your/path/zhongliu_checkpoint.pth')
   ```

## 关键参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `img_size` | 256 | 输入/输出图像尺寸 |
| `batch_size` | 8 | 训练与测试批量大小 |
| `lr` | 0.005 | Adam 学习率 |
| `epochs` | 10 | 训练轮次 |
| `patience` | 10 | 早停耐心值 |
| `monitor` | val_loss | 早停监控指标 |

## 备注

- 输入为单通道灰度图，输出为单通道 logits，预测时通过 `Sigmoid` 转为概率。
- 数据路径示例位于 `E:\Vscode\jupyter\data\zhongliu\...`，在 macOS / Linux 上运行时需要同步修改。
- 当前脚本未做训练/验证/测试集划分与数据增强，可按需扩展。
