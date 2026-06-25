import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import sys

sys.path.append('.')  # 确保可以导入同级模块

import config
from data_loader import get_train_val_loaders
from model import ViTClassifier
from utils import save_checkpoint, load_checkpoint, plot_training_curves


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 数据加载
    train_loader, val_loader, class_names = get_train_val_loaders()
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # 模型、损失函数、优化器
    model = ViTClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # 训练记录
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc,
                            os.path.join(config.SAVE_DIR, 'best.pth'))
            print(f"Best model saved with val_acc: {val_acc:.2f}%")

    # 保存最后模型
    save_checkpoint(model, optimizer, config.EPOCHS, val_acc,
                    os.path.join(config.SAVE_DIR, 'last.pth'))

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    print("Training completed!")


if __name__ == "__main__":
    main()