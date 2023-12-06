from tqdm import tqdm
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


device = torch.device('cuda:0')


def dataloader_1():
    # 使用torchvision的预定义数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='../dataset/', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # tqdm
    nb = len(train_loader)
    pbar = enumerate(train_loader)
    pbar = tqdm(pbar, total=nb)
    for batch_idx, data in pbar:
        time.sleep(0.1)
        input_batch = data[0].to(device, non_blocking=True)
        label_batch = data[1].to(device, non_blocking=True)
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)    # 当前已占用的显存
        pbar.set_description('gpu reserved: {mem}'.format(mem=mem))


def dataloader_2():
    # 自定义数据集，加载时使用多进程(num_workers)存在问题
    class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            sample = self.data[item]
            label = self.labels[item]
            return sample, label

    my_data = torch.randn(100, 3, 32, 32)
    my_labels = torch.randint(0, 10, (100,))
    dataset = CustomDataset(my_data, my_labels)
    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    # tqdm
    nb = len(loader)
    pbar = enumerate(loader)
    pbar = tqdm(pbar, total=nb)
    for batch_idx, data in pbar:
        time.sleep(0.1)
        input_batch = data[0].to(device, non_blocking=True)
        label_batch = data[1].to(device, non_blocking=True)
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)    # 当前已占用的显存
        pbar.set_description('gpu reserved: {mem}'.format(mem=mem))


if __name__ == "__main__":
    dataloader_2()
    print('done')













