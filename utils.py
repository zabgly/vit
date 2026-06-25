import torch
import matplotlib.pyplot as plt
import os


def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """保存检查点"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """加载检查点"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['accuracy']


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()
    print("Training curves saved to training_curves.png")