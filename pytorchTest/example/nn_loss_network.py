# 将图片转换为tensor格式
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from models.vgg_tudui import Tudui

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 获取pytorch官网上的数据集，下载到本地，转为tensor格式
dataset = torchvision.datasets.CIFAR10(root="../dataset", transform=dataset_transform, train=False, download=True)

dataloader = DataLoader(dataset, batch_size=1)

loss = nn.CrossEntropyLoss()
tudui = Tudui()

# 优化器
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)

# 学习20轮
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        # 每次的梯度清零
        optim.zero_grad()
        result_loss.backward()
        optim.step() 
        running_loss = running_loss + result_loss
    print(running_loss)

