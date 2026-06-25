import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import config

def get_transform(mode='train'):
    """根据模式获取数据增强管道"""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(*config.TRAIN_TRANSFORMS['color_jitter']),
            transforms.ToTensor(),
            transforms.Normalize(**config.TRAIN_TRANSFORMS['normalize'])
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(**config.TRAIN_TRANSFORMS['normalize'])
        ])

def get_dataloader(data_dir, batch_size=config.BATCH_SIZE, mode='train', shuffle=True):
    """创建数据加载器"""
    transform = get_transform(mode)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    return dataloader, dataset.classes

# 示例：创建训练和验证加载器
def get_train_val_loaders():
    train_loader, classes = get_dataloader(config.TRAIN_DIR, mode='train', shuffle=True)
    val_loader, _ = get_dataloader(config.VAL_DIR, batch_size=config.BATCH_SIZE, mode='val', shuffle=False)
    return train_loader, val_loader, classes