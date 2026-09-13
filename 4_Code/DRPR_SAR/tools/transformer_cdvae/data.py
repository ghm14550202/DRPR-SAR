import os
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
DATASET_DIRS = {
    "SOC_40": "SOC_40classes",
    "SOC_40classes": "SOC_40classes",
    "MSTAR": "mstar",
    "mstar": "mstar",
    "FUSAR": "FUSAR",
    "fusar": "FUSAR",
}


def resolve_data_path(dataset, data_root=DATA_ROOT, data_path=None):
    if data_path:
        return Path(data_path).expanduser()
    if dataset not in DATASET_DIRS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return Path(data_root).expanduser() / DATASET_DIRS[dataset]


def image_transform(size=224, train=True):
    ops = [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((size, size)),
    ]
    if train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def load_datasets(dataset, data_root=DATA_ROOT, data_path=None, size=224):
    data_path = resolve_data_path(dataset, data_root, data_path)
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"Dataset must contain train/ and test/: {data_path}")

    train_set = datasets.ImageFolder(str(train_dir), transform=image_transform(size, train=True))
    test_set = datasets.ImageFolder(str(test_dir), transform=image_transform(size, train=False))

    remapped_samples = []
    missing = set()
    for path, local_idx in test_set.samples:
        class_name = test_set.classes[local_idx]
        if class_name not in train_set.class_to_idx:
            missing.add(class_name)
            continue
        remapped_samples.append((path, int(train_set.class_to_idx[class_name])))
    if missing:
        raise ValueError(f"Test classes not found in train set: {sorted(missing)}")
    test_set.samples = remapped_samples
    test_set.targets = [target for _, target in remapped_samples]
    test_set.class_to_idx = dict(train_set.class_to_idx)

    return data_path, train_set, test_set


def build_loaders(dataset, batch_size, data_root=DATA_ROOT, data_path=None, size=224, num_workers=4):
    data_path, train_set, test_set = load_datasets(dataset, data_root, data_path, size)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return data_path, train_set, test_set, train_loader, test_loader
