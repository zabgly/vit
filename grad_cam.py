import argparse
import os
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

import config
from model import ViTClassifier
from data_loader import get_transform


def load_model(model_path: str, device: torch.device) -> nn.Module:
    model = ViTClassifier(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device).eval()
    return model


def find_last_conv2d(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._h1 = target_layer.register_forward_hook(self._forward_hook)
        # full_backward_hook 在某些版本更稳，这里做兼容
        if hasattr(target_layer, "register_full_backward_hook"):
            self._h2 = target_layer.register_full_backward_hook(self._backward_hook)
        else:
            self._h2 = target_layer.register_backward_hook(self._backward_hook)  # type: ignore[attr-defined]

    def close(self):
        self._h1.remove()
        self._h2.remove()

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> tuple[torch.Tensor, int]:
        """
        返回:
          cam: [H, W] 0~1
          class_idx: 使用的类别索引
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)  # [1, C]
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hook 未捕获到 activation/gradient；请检查 target_layer 是否正确。")

        # activations/grads: [B, K, h, w]
        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, K, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)  # [B,1,H,W]
        cam = cam[0, 0]
        cam = self._normalize_cam(cam)
        return cam, class_idx


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: [3,H,W] after Normalize
    return: uint8 RGB [H,W,3]
    """
    mean = torch.tensor(config.TRAIN_TRANSFORMS["normalize"]["mean"], device=img_tensor.device)[:, None, None]
    std = torch.tensor(config.TRAIN_TRANSFORMS["normalize"]["std"], device=img_tensor.device)[:, None, None]
    img = img_tensor * std + mean
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return img


def overlay_cam(rgb: np.ndarray, cam01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    rgb: uint8 [H,W,3]
    cam01: float [H,W] 0~1
    """
    heat = plt.get_cmap("jet")(cam01)[..., :3]  # [H,W,3] 0~1
    heat = (heat * 255.0).astype(np.uint8)
    out = (rgb * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM 可视化（适用于包含Conv2d的timm/efficientvit等backbone）")
    parser.add_argument("--image", type=str, required=True, help="单张图片路径")
    parser.add_argument("--ckpt", type=str, default="", help="模型权重路径（默认自动找best.pth/last.pth）")
    parser.add_argument("--class-idx", type=int, default=-1, help="指定类别索引（默认用预测类别）")
    parser.add_argument("--out", type=str, default="grad_cam.png", help="输出图片路径")
    args = parser.parse_args()

    device = torch.device(config.DEVICE)

    # checkpoint 路径
    ckpt = args.ckpt
    if not ckpt:
        best = os.path.join(config.SAVE_DIR, "best.pth")
        last = os.path.join(config.SAVE_DIR, "last.pth")
        ckpt = best if os.path.exists(best) else last
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"未找到checkpoint: {ckpt}")

    model = load_model(ckpt, device)

    # 尝试优先在 backbone 内找最后一个 Conv2d；找不到再退回整个模型
    target = find_last_conv2d(getattr(model, "backbone", model)) or find_last_conv2d(model)
    if target is None:
        raise RuntimeError(
            "当前模型里找不到 nn.Conv2d，无法用经典Grad-CAM。\n"
            "如果你用的是纯Transformer(ViT/DeiT)结构，需要用 Attention Rollout / Transformer-CAM 方案。"
        )

    cam_engine = GradCAM(model, target)

    # 读图 + 预处理
    img_pil = Image.open(args.image).convert("RGB")
    tfm = get_transform("test")
    x = tfm(img_pil).unsqueeze(0).to(device)  # [1,3,H,W]

    class_idx = None if args.class_idx < 0 else int(args.class_idx)
    cam, used_idx = cam_engine(x, class_idx=class_idx)
    cam_engine.close()

    rgb = denormalize(x[0])
    cam01 = cam.detach().cpu().numpy()
    out = overlay_cam(rgb, cam01, alpha=0.45)

    # 保存
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.axis("off")
    plt.imshow(rgb)

    plt.subplot(1, 3, 2)
    plt.title(f"Grad-CAM (class={used_idx})")
    plt.axis("off")
    plt.imshow(cam01, cmap="jet")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.axis("off")
    plt.imshow(out)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out} (class={used_idx}, ckpt={ckpt})")


if __name__ == "__main__":
    main()


