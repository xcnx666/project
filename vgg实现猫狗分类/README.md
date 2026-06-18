# VGG 猫狗分类项目操作文档

> 基于 VGG16 预训练模型的迁移学习实践项目，使用 PyTorch 完成猫 / 狗二分类。

---

## 1. 项目简介

本项目以 ImageNet 预训练的 **VGG16** 模型为基础，采用**迁移学习**方法对 Kaggle "Dogs vs. Cats" 数据集进行二分类。

- **任务类型**：图像二分类（猫 / 狗）
- **核心模型**：`torchvision.models.vgg16`
- **训练策略**：冻结卷积层（特征提取器），仅微调分类器的最后两层
- **训练框架**：`torchkeras`（封装 PyTorch，类 Keras 训练循环）
- **指标库**：`torchmetrics`

---

## 2. 目录结构

```
vgg实现猫狗分类/
├── README.md
└── vgg猫狗分类/
    ├── vgg.ipynb                # 主 Notebook（训练 + 预测）
    └── models/                  # 需自行创建，存放权重
        ├── vgg16-397923af.pth   # 预训练权重（需自行下载）
        └── my_vgg               # 微调后保存的模型
```

数据目录（需自行准备，路径在 Notebook 中可改）：

```
data/
└── dogs_vs_cats_sample/
    ├── train/
    │   ├── cats/   # 猫的训练图片
    │   └── dogs/   # 狗的训练图片
    └── test1/      # 测试图片（无标签，用于可视化预测）
```

---

## 3. 环境准备

### 3.1 推荐版本

| 组件 | 版本建议 |
| --- | --- |
| Python | 3.8 ~ 3.10 |
| PyTorch | 1.10 ~ 2.x（与 CUDA 版本匹配） |
| torchvision | 与 PyTorch 对应版本 |
| torchkeras | 最新版 |
| torchmetrics | ≥ 0.10 |
| torchsummary | 任意 |
| matplotlib | 任意 |

### 3.2 安装命令

```bash
pip install torch torchvision
pip install torchkeras torchmetrics torchsummary matplotlib
```

> Apple Silicon (M1/M2) 用户：参考 PyTorch 官网安装带 MPS 后端的版本，代码中 `cpu=False` 会自动使用 GPU / MPS。

### 3.3 预训练权重

从 PyTorch 官方或 torchvision 缓存获取 `vgg16-397923af.pth`，放入 `vgg猫狗分类/models/` 目录。

也可由代码自动下载（首次调用 `models.vgg16(weights=None)` 后改为 `VGG16_Weights.DEFAULT`）。

---

## 4. 数据准备

1. 下载 Dogs vs. Cats 数据集（Kaggle）。
2. 在项目根目录创建 `data/dogs_vs_cats_sample/`，按下列结构组织：
   - `train/cats/*.jpg`
   - `train/dogs/*.jpg`
   - `test1/*.jpg`（无类别子目录，顺序由字母 / 数字决定）
3. ⚠️ 训练集和验证集（test1）建议由自己按比例切分，本项目为了简化直接将 `test1` 作为验证集。

---

## 5. Notebook 核心流程

打开 `vgg猫狗分类/vgg.ipynb`，按顺序执行以下单元：

### 5.1 数据加载与增强

```python
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

train_dataset = ImageFolder(root='data/dogs_vs_cats_sample/train', transform=transform)
val_dataset   = ImageFolder(root='data/dogs_vs_cats_sample/test1', transform=transform)
```

- 输入尺寸统一为 **224×224**（VGG16 默认）。
- 归一化到 `[-1, 1]`。
- `ImageFolder` 会按文件夹名排序映射类别：本项目 `classes = ['cat', 'dog']`。

### 5.2 DataLoader

```python
train_dataload = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataload   = DataLoader(val_dataset,   batch_size=8, shuffle=False)
```

### 5.3 模型构建（迁移学习关键步骤）

```python
model = models.vgg16(weights=None)
model.load_state_dict(torch.load('models/vgg16-397923af.pth'))

# 1) 冻结全部卷积层与原分类器
for param in model.parameters():
    param.requires_grad = False

# 2) 替换分类器为二分类头
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(4096,  4096), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(4096,  2)
)

# 3) 仅解冻最后一层 (index 4, 5)
for index, param in enumerate(model.classifier.parameters()):
    if index == 4 or index == 5:
        param.requires_grad = True
```

> 通过 `summary(model, (3,224,224))` 可查看可训练参数量，确认只有最后一层 Linear 被更新。

### 5.4 模型训练

```python
vgg_model = torchkeras.KerasModel(
    model,
    loss_fn   = nn.CrossEntropyLoss(),
    optimizer = torch.optim.Adam(model.classifier.parameters()),
    metrics_dict = {'acc': torchmetrics.Accuracy(task='multiclass', num_classes=2)}
)

vgg_history = vgg_model.fit(
    train_data = train_dataload,
    val_data   = val_dataload,
    epochs     = 10,
    plot       = True,
    cpu        = False
)
```

训练完成后：
- 控制台输出每轮 `train_loss / val_loss / train_acc / val_acc`。
- `plot=True` 自动绘制曲线。
- 历史记录可通过 `vgg_history` 进一步可视化（见 Notebook 中 `plt.subplot(121/122)` 单元）。

### 5.5 模型保存

```python
torch.save(model.state_dict(), 'models/my_vgg')
```

### 5.6 推理与可视化

```python
# 重建相同结构的模型
my_vgg_model = vgg16(weights=False)
my_vgg_model.classifier = nn.Sequential(...)   # 同 5.3
my_vgg_model.load_state_dict(torch.load('models/my_vgg'))

# 预测
test_loader = DataLoader(test_img, batch_size=8)
img_1, _    = next(iter(test_loader))
pred        = my_vgg_model(img_1)
pred_label  = torch.argmax(torch.softmax(pred, dim=1), dim=1)

classes = ['cat', 'dog']
print([classes[i] for i in pred_label])
```

最后一段 `make_grid + imshow` 将批量图片与预测标签一起展示。

---

## 6. 常见操作清单

| 操作 | 做法 |
| --- | --- |
| 换数据集 | 修改 `train_dataset / val_dataset` 的 `root`，保持子目录结构（`class_name/*.jpg`） |
| 改输入尺寸 | 同时修改 `CenterCrop(size)` 与 `summary(input_size=...)` |
| 调整训练强度 | 解冻更多层：去除 `if index == 4 or index == 5` 的限制 |
| 改优化器 | 替换 `torch.optim.Adam(...)` 为 SGD / AdamW 等 |
| 多分类 | 把 `nn.Linear(4096, 2)` 的 `2` 改为类别数，并修改 `num_classes` |
| 启用 GPU | `model.cuda()`，数据 `.cuda()`，或直接交给 `torchkeras` 自动处理 |

---

## 7. 注意事项与改进建议

1. **路径硬编码问题**  
   Notebook 中存在 Windows 绝对路径（如 `r'C:\Users\39608\Desktop\peoject\data\dogs_vs_cats_sample'`）。在 macOS / Linux 上运行请改为相对路径或本机路径。

2. **验证集与测试集混用**  
   `test1` 同时被当作验证集和推理可视化源。正式项目应划分独立的验证集，并将 `test1` 仅用于推理。

3. **类别映射**  
   `ImageFolder` 按字母序排序，因此 `cats → 0, dogs → 1`。若类别名变化，记得同步修改 `classes = [...]`。

4. **数据增强偏弱**  
   训练阶段只用了 `CenterCrop`，建议加入 `RandomHorizontalFlip`、`RandomResizedCrop` 提升泛化能力。

5. **batch_size 过小**  
   `batch_size=8` 在显存允许时可调到 32 / 64，加速收敛。

6. **仅微调最后一层**  
   若数据集更大、想获得更高精度，可解冻 `classifier` 的全部 6 个参数（或最后几个卷积块），并降低学习率。

7. **模型保存**  
   `torch.save(model.state_dict(), ...)` 是推荐方式；加载时务必**先重建结构**再 `load_state_dict`，否则会报 `unexpected keys` 错误。

---

## 8. 故障排查

| 现象 | 可能原因 | 解决方案 |
| --- | --- | --- |
| `FileNotFoundError: vgg16-397923af.pth` | 权重路径不对 | 确认 `models/` 目录存在并放入权重 |
| `RuntimeError: Error(s) in loading state_dict` | 分类器未替换 / 替换结构不一致 | 严格按 5.3 步骤重建 `classifier` |
| `num_classes` 报错 | `torchmetrics.Accuracy` 未指定类别数 | 传入 `num_classes=2` |
| 训练时 Loss 不下降 | 学习率太大 / 数据未归一化 | 减小 LR、确认 `Normalize` 生效 |
| 推理时全预测为同一类 | 数据未 shuffle、模型未加载权重 | 打印模型参数 `requires_grad`、打印 logits |
| macOS MPS 报错 | 部分算子不支持 | 设 `cpu=False` 但实际回退到 CPU；或升级 PyTorch |

---

## 9. 复现步骤一览

```bash
# 1. 准备环境
pip install torch torchvision torchkeras torchmetrics torchsummary matplotlib

# 2. 准备数据与权重
mkdir -p data/dogs_vs_cats_sample models
# 将 cats/dogs 图片放入 train/{cats,dogs}
# 将测试图片放入 test1/
# 将 vgg16-397923af.pth 放入 models/

# 3. 启动 Notebook
cd vgg猫狗分类
jupyter notebook vgg.ipynb
```

依次执行所有 Cell，训练结束后即可在 `models/my_vgg` 看到保存的权重。

---

## 10. 版本记录

| 日期 | 变更 | 备注 |
| --- | --- | --- |
| 初始版本 | 完成迁移学习训练与推理 | 10 epochs，batch_size=8，仅解冻最后一层 |

---

如需进一步扩展（如：部署为 Flask 服务、转为 ONNX、加入 Grad-CAM 可视化等），可在本项目基础上继续迭代。
