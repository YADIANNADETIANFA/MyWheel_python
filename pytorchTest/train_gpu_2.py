import torch.optim
import torchvision.datasets
import time
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models.vgg_tudui import Tudui

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

# 定义训练的设备
# device = torch.device("cpu")
# device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10(root='./dataset', transform=dataset_transform, train=True, download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset', transform=dataset_transform, train=False, download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

# 利用DataLoader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()
# ---------cuda-------------
tudui = tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# ---------cuda-------------
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs_train")

start_time = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练步骤开始
    # 将网络设置成训练模式
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        # ---------cuda-------------
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    # 将网络设置成测试模式
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    # 测试网络时取消梯度属性
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # ---------cuda-------------
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            # outputs.argmax(1) 横向取识别判断概率最大的值
            # sun() 将所有64个True=1，False=0的结果加和
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "./trained_models/tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()

# 亦可使用Google免费GPU
















