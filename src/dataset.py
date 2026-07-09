import ssl
import certifi
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


def get_mnist_loaders(data_dir: str, batch_size: int, num_workers: int = 2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
