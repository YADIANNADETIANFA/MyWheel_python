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


def dataloader_3():
    """
    collate_fn是DataLoader类的一个参数，用于指定如何将多个数据样本整合到一个batch中。
    即在每个epoch中迭代DataLoader来获取batch时，collate_fn将在这个batch中被调用，将数据样本列表(从Dataset.__getitem__返回的)转换为DataLoader的输出批次格式。
    """
    # 自定义一个简单的Dataset
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            return self.data[item]

    # 自定义collate_fn函数，将样本列表合并成一个batch
    def my_collate_fn(batch):
        # 在这个例子中，我们简单地把样本合并成一个列表
        return batch

    # 创建一个数据集实例
    data = list(range(100))
    dataset = MyDataset(data)

    # 创建一个DataLoader实例，并指定collate_fn
    data_loader = DataLoader(dataset=dataset, batch_size=4, collate_fn=my_collate_fn)

    # 迭代DataLoader来获取batch
    for batch in data_loader:
        print(batch)


if __name__ == "__main__":
    dataloader_3()
    print('done')













