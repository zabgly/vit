import torch
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('.')

import config
from model import ViTClassifier
from data_loader import get_transform


def load_model(model_path, device):
    model = ViTClassifier(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def validate(model, dataloader, device, class_names):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 计算准确率
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Overall Accuracy: {accuracy:.2f}%")

    # 分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
    print("Confusion matrix saved to confusion_matrix.png")

    return accuracy


def main():
    device = torch.device(config.DEVICE)

    # 准备验证数据
    transform = get_transform('val')
    val_dataset = datasets.ImageFolder(root=config.VAL_DIR, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    class_names = val_dataset.classes

    # 加载最佳模型
    model_path = os.path.join(config.SAVE_DIR, 'best.pth')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, using last.pth instead.")
        model_path = os.path.join(config.SAVE_DIR, 'last.pth')

    model = load_model(model_path, device)
    print(f"Model loaded from {model_path}")

    # 验证
    accuracy = validate(model, val_loader, device, class_names)


if __name__ == "__main__":
    main()