from typing import Tuple

import torchvision.datasets as dset
import torchvision.transforms as transforms

def build_dataset(
    data_dir: str
) -> Tuple[dset.CIFAR100, dset.CIFAR100, dset.CIFAR100]:
    """
    Builds CIFAR-100 train, validation, and test datasets.
    - data_dir: path to CIFAR-100 folder.
    """
    # normalization stats
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    # transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # datasets
    train_dataset = dset.CIFAR100(root=data_dir, train=True,  download=False,  transform=train_transform)
    val_dataset   = dset.CIFAR100(root=data_dir, train=True,  download=False,  transform=test_transform)
    test_dataset  = dset.CIFAR100(root=data_dir, train=False, download=False,  transform=test_transform)

    return train_dataset, val_dataset, test_dataset