import torch.nn as nn
import timm
import config


class ViTClassifier(nn.Module):
    """ViT分类模型（支持加载预训练权重）"""

    def __init__(self, model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained=True):
        super().__init__()
        # 加载timm中的ViT模型，去掉分类头
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # 获取特征维度（ViT-base通常为768）
        if 'efficientvit' in model_name:
            feature_dim = 224
        elif 'small' in model_name:
            feature_dim = 384
        elif 'base' in model_name:
            feature_dim = 768
        elif 'large' in model_name:
            feature_dim = 1024
        else:
            feature_dim = 768  # 默认

        # 自定义分类头（含Dropout防止过拟合）
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)  # [B, feature_dim]
        output = self.classifier(features)
        return output


# 测试模型
if __name__ == "__main__":
    model = ViTClassifier()
    print(model)
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")