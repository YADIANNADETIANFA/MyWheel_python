import torchvision

# train_data = torchvision.datasets.ImageNet("./data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

from torch import nn

# 未训练的vgg的网络模型，分类网络
vgg16_false = torchvision.models.vgg16(pretrained=False)

# 已训练好的vgg网络模型，分类网络
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_data = torchvision.datasets.CIFAR10(root="../dataset", transform=dataset_transform, train=True, download=True)

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)


