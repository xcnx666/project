"""
肿瘤图像分割
=============

本脚本基于 U-Net 实现肿瘤图像分割，包含数据加载、模型定义、训练、模型导入和预测可视化。
"""

# %% [markdown]
# ### Dataset类
# 图像分割的特征是图片,标签也是图片,需要在原始的Dataset类上构建新的dataset类

# %%
from PIL import Image

# %%
x = Image.open(r'img_path')

# %% [markdown]
# ### Dataset类
# 图像分割的特征是图片,标签也是图片,需要在原始的Dataset类上构建新的dataset类

# %%
from torch.utils.data import Dataset

# %%
from glob import glob

# %%
glob(r'E:\Vscode\jupyter\data\zhongliu\tumor_data\train\patient\*')[:3]

# %%
class LoadDataset(Dataset):
    def __init__(self, img_root, mask_root, transform=None, target_transform=None):
        self.imgs_file = glob(img_root)
        self.mask_file = glob(mask_root)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs_file[index]
        y_path = self.mask_file[index]

        img_x = Image.open(x_path).convert('L')
        img_y = Image.open(y_path).convert('L')

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs_file)

# %%
from torchvision import transforms

# %%
img_size = 256

# %%
img_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ]
)

# %%
target_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ]
)

# %%
# 构建示例
train_img_root = 'E:/Vscode/jupyter/data/zhongliu/tumor_data/train/patient/*'
train_mask_root = 'E:/Vscode/jupyter/data/zhongliu/tumor_data/train/tumor/*'
test_img_root = 'E:/Vscode/jupyter/data/zhongliu/tumor_data/test/patient/*'
test_mask_root = 'E:/Vscode/jupyter/data/zhongliu/tumor_data/test/tumor/*'

# %%
train_dataset = LoadDataset(train_img_root, train_mask_root, transform=img_transform, target_transform=target_transform)
test_dataset = LoadDataset(test_img_root, test_mask_root, transform=img_transform, target_transform=target_transform)

# %%
for i, j in train_dataset:
    print(i.shape, j.shape)
    break

# %% [markdown]
# ### 构建DataLoader类

# %%
from torch.utils.data import DataLoader

# %%
batch_size = 8

# %%
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# %%
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# %%
for i, j in train_loader:
    print(i.shape, j.shape)
    break

# %% [markdown]
# ### 搭建模型

# %% [markdown]
# ##### batch normalization批量正则化 与 layer normalization层正则化的区别

# %%
from torch.nn import Sequential, Conv2d, BatchNorm2d, Module, ReLU, MaxPool2d, ConvTranspose2d

# %%
class Conv(Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.feature = Sequential(
            Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),  # 数据维度(1,256,256) -> (64,256,256)
            BatchNorm2d(out_channel),
            ReLU(),
            Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),  # 数据维度(64,256,256) -> (64,256,256)
            BatchNorm2d(out_channel),
            ReLU()
        )

    def forward(self, x):
        return self.feature(x)

# %%
import torch

class Unet(Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(Unet, self).__init__()
        self.conv1 = Conv(in_channel=in_channel, out_channel=64)
        self.conv2 = Conv(in_channel=64, out_channel=128)
        self.conv3 = Conv(128, 256)
        self.conv4 = Conv(256, 512)
        self.conv5 = Conv(512, 1024)

        self.pool = MaxPool2d(2)

        self.up1 = ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv6 = Conv(1024, 512)

        self.up2 = ConvTranspose2d(512, 256, 2, 2)
        self.conv7 = Conv(512, 256)

        self.up3 = ConvTranspose2d(256, 128, 2, 2)
        self.conv8 = Conv(256, 128)

        self.up4 = ConvTranspose2d(128, 64, 2, 2)
        self.conv9 = Conv(128, 64)

        self.conv10 = Conv2d(64, out_channel, 3, 1, 1)

    def forward(self, x):
        xc1 = self.conv1(x)
        xp1 = self.pool(xc1)

        xc2 = self.conv2(xp1)
        xp2 = self.pool(xc2)

        xc3 = self.conv3(xp2)
        xp3 = self.pool(xc3)

        xc4 = self.conv4(xp3)
        xp4 = self.pool(xc4)

        xc5 = self.conv5(xp4)

        xu1 = self.up1(xc5)
        xm1 = torch.cat([xc4, xu1], dim=1)  # 特征融合
        xc6 = self.conv6(xm1)  # 融合后需要再做一次卷积操作

        xu2 = self.up2(xc6)
        xm2 = torch.cat([xc3, xu2], dim=1)
        xc7 = self.conv7(xm2)

        xu3 = self.up3(xc7)
        xm3 = torch.cat([xc2, xu3], dim=1)
        xc8 = self.conv8(xm3)

        xu4 = self.up4(xc8)
        xm4 = torch.cat([xc1, xu4], dim=1)
        xc9 = self.conv9(xm4)

        xc10 = self.conv10(xc9)

        return xc10

# %%
from torchkeras import KerasModel

# %%
net = Unet()

# %%
model = KerasModel(
    net,
    loss_fn=torch.nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam(net.parameters(), lr=0.005)
)

# %%
import torchsummary

# %%
torchsummary.summary(model, input_size=(1, img_size, img_size), device='cpu')

# %%
ckpt_path = 'E:/Vscode/jupyter/data/zhongliu/zhongliu_checkpooiny'

# %%
history = model.fit(
    train_data=train_loader,
    val_data=test_loader,
    patience=10,
    epochs=10,
    monitor='val_loss',
    mode='min',
    ckpt_path=ckpt_path,
    plot=True,
    cpu=False,
)

# %%
history

# %%
# torch.save(model.state_dic(),'E:/Vscode/jupyter/data/zhongliu/zhongliu_checkpooiny.pth')

# %% [markdown]
# # 模型的导入

# %%
# 加载训练好的模型
import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, Module, ReLU, MaxPool2d, ConvTranspose2d

# %%
class Conv(Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.feature = Sequential(
            Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channel),
            ReLU(),
            Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channel),
            ReLU()
        )

    def forward(self, x):
        return self.feature(x)

import torch

class Unet(Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(Unet, self).__init__()
        self.conv1 = Conv(in_channel=in_channel, out_channel=64)
        self.conv2 = Conv(in_channel=64, out_channel=128)
        self.conv3 = Conv(128, 256)
        self.conv4 = Conv(256, 512)
        self.conv5 = Conv(512, 1024)

        self.pool = MaxPool2d(2)

        self.up1 = ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv6 = Conv(1024, 512)

        self.up2 = ConvTranspose2d(512, 256, 2, 2)
        self.conv7 = Conv(512, 256)

        self.up3 = ConvTranspose2d(256, 128, 2, 2)
        self.conv8 = Conv(256, 128)

        self.up4 = ConvTranspose2d(128, 64, 2, 2)
        self.conv9 = Conv(128, 64)

        self.conv10 = Conv2d(64, out_channel, 3, 1, 1)

    def forward(self, x):
        xc1 = self.conv1(x)
        xp1 = self.pool(xc1)

        xc2 = self.conv2(xp1)
        xp2 = self.pool(xc2)

        xc3 = self.conv3(xp2)
        xp3 = self.pool(xc3)

        xc4 = self.conv4(xp3)
        xp4 = self.pool(xc4)

        xc5 = self.conv5(xp4)

        xu1 = self.up1(xc5)
        xm1 = torch.cat([xc4, xu1], dim=1)
        xc6 = self.conv6(xm1)

        xu2 = self.up2(xc6)
        xm2 = torch.cat([xc3, xu2], dim=1)
        xc7 = self.conv7(xm2)

        xu3 = self.up3(xc7)
        xm3 = torch.cat([xc2, xu3], dim=1)
        xc8 = self.conv8(xm3)

        xu4 = self.up4(xc8)
        xm4 = torch.cat([xc1, xu4], dim=1)
        xc9 = self.conv9(xm4)

        xc10 = self.conv10(xc9)

        return xc10

# %%
from torchkeras import KerasModel
net = Unet()
model = KerasModel(
    net,
    loss_fn=torch.nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam(net.parameters(), lr=0.005)
)

# %%
model.load_state_dict(torch.load(r'E:\Vscode\jupyter\data\zhongliu\tumor_checkpoint_260416.pth'))

# %% [markdown]
# ### 加载和预处理

# %%
from torch.utils.data import dataset
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

# %%
class LoadDataset(Dataset):
    def __init__(self, img_root, mask_root, transform=None, target_transform=None):
        self.imgs_file = glob(img_root)
        self.mask_file = glob(mask_root)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs_file[index]
        y_path = self.mask_file[index]

        img_x = Image.open(x_path).convert('L')
        img_y = Image.open(y_path).convert('L')

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs_file)

# %%
from torchvision import transforms
img_size = 256
img_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ]
)

target_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ]
)
test_img_root = 'E:/Vscode/jupyter/data/zhongliu/tumor_data/test/patient/*'
test_mask_root = 'E:/Vscode/jupyter/data/zhongliu/tumor_data/test/tumor/*'
test_dataset = LoadDataset(test_img_root, test_mask_root, transform=img_transform, target_transform=target_transform)

# %%
for i, j in test_dataset:
    print(i.shape, j.shape)
    break

# %% [markdown]
# #### 模型预测

# %%
import matplotlib.pyplot as plt

# %%
for i, (x, y) in enumerate(test_dataset):
    plt.figure(figsize=(12, 10))
    plt.subplot(141)
    plt.imshow(x[0], cmap='gray')
    plt.title('img')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(y[0], cmap='gray')
    plt.title('mask')
    plt.axis('off')

    x = torch.unsqueeze(x, dim=0)
    pre = model.forward(x)

    sig = torch.nn.Sigmoid()
    pre = sig(pre)

    plt.subplot(143)
    plt.imshow(pre.detach().numpy()[0, 0, :, :], cmap='gray')
    plt.title('mode_pre')
    plt.axis('off')

    plt.show()
    if i > 10:
        break