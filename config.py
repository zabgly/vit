
import os
import torch
# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')           # 数据集根目录
TRAIN_DIR = os.path.join(DATA_DIR, 'train')         # 训练集
VAL_DIR = os.path.join(DATA_DIR, 'val')             # 验证集
TEST_DIR = os.path.join(DATA_DIR, 'test')           # 测试集（可选）

# 模型参数
MODEL_NAME = 'efficientvit_m2'                  # ViT模型名称（timm库）
NUM_CLASSES = 4                                       # 建筑阶段数
INPUT_SIZE = 224                                      # 输入图像尺寸

# 训练参数
BATCH_SIZE = 16                                       # 批次大小（根据GPU内存调整）
EPOCHS = 50                                           # 训练轮次
LEARNING_RATE = 1e-4                                  # 初始学习率
WEIGHT_DECAY = 1e-4                                    # 权重衰减
NUM_WORKERS = 4                                        # 数据加载线程数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据增强参数
TRAIN_TRANSFORMS = {
    'resize': INPUT_SIZE,
    'random_horizontal_flip': True,
    'random_vertical_flip': True,
    'random_rotation': 30,
    'color_jitter': (0.2, 0.2, 0.2, 0.1),
    'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
}

# 保存路径
SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)