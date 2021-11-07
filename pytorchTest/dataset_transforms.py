import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 将图片转换为tensor格式
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 获取pytorch官网上的数据集，下载到本地，转为tensor格式
train_set = torchvision.datasets.CIFAR10(root="./dataset", transform=dataset_transform, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", transform=dataset_transform, train=False, download=True)

# writer = SummaryWriter("p10")
# for i in range(10):
#     img, target = test_set[i]
#     writer.add_image("test_set", img, i)
# writer.close()


# batch_size=4：单次获取64张图片；
# shuffle=False：不打乱数据集，每次都已相同的顺序返回
# num_workers=0：单进程取图片
# drop_last=False：单次不足64张图片时依然取出，不舍弃
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer2 = SummaryWriter("dataset_transforms_loader")
step = 0
for data in test_loader:
    imgs, targets = data
    # add_images,不是add_image
    writer2.add_images("dataset_transforms_test", imgs, step)
    step = step + 1
writer2.close()
