import torchvision
from torch.utils.data import DataLoader

train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),  # 转换成张量
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                       ]))

train_loader = DataLoader(train, batch_size=2)

for data in train_loader:
    print(data[0].shape)
    break