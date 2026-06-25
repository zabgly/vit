import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import ViTClassifier  # 确保 model.py 在同一目录


def load_model(ckpt_path, device, model_name='efficientvit_m2', num_classes=4):
    """加载训练好的模型"""
    model = ViTClassifier(model_name=model_name, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 兼容不同的保存格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def get_transform(input_size=224):
    """与训练时一致的预处理"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def predict_image(model, img_path, transform, device, class_names):
    """预测单张图片，返回 (类别索引, 类别名, 置信度)"""
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_idx].item()

    return pred_idx, class_names[pred_idx], conf


def main():
    parser = argparse.ArgumentParser(description='ViT建筑阶段分类推理（终端打印）')
    parser.add_argument('--ckpt', required=True, help='模型权重文件路径')
    parser.add_argument('--image', help='单张图片路径')
    parser.add_argument('--image_dir', help='批量图片文件夹路径')
    parser.add_argument('--model_name', default='efficientvit_m2', help='模型名称')
    parser.add_argument('--num_classes', type=int, default=4, help='类别数')
    parser.add_argument('--input_size', type=int, default=224, help='输入尺寸')
    parser.add_argument('--device', default='cuda', help='设备 (cuda/cpu)')
    args = parser.parse_args()

    # 设备设置
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f'使用设备: {device}')

    # 类别名称（根据实际修改）
    class_names = ['Stage1: 基础施工', 'Stage2: 主体结构', 'Stage3: 封顶装修', 'Stage4: 竣工清理']

    # 加载模型
    print(f'加载模型: {args.ckpt}')
    model = load_model(args.ckpt, device, args.model_name, args.num_classes)
    transform = get_transform(args.input_size)

    # 单张推理
    if args.image:
        idx, pred, conf = predict_image(model, args.image, transform, device, class_names)
        print(f'\n图片: {args.image}')
        print(f'预测类别: {pred} (索引 {idx})')
        print(f'置信度: {conf:.4f}')

    # 批量推理
    elif args.image_dir:
        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        images = [f for f in os.listdir(args.image_dir) if f.lower().endswith(exts)]
        if not images:
            print(f'文件夹 {args.image_dir} 中没有图片文件')
            return
        print(f'\n共 {len(images)} 张图片，开始推理...')
        for img_file in images:
            img_path = os.path.join(args.image_dir, img_file)
            idx, pred, conf = predict_image(model, img_path, transform, device, class_names)
            print(f'{img_file}: {pred} (置信度 {conf:.2%})')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()