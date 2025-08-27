import argparse
import os
import random
import time
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")

# ---- Utils ----
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_device():
    print("正在檢查與設定設備...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"成功偵測到 CUDA 設備！將使用 GPU 進行訓練：{torch.cuda.get_device_name(0)}")
        # 明確指定使用第一個 CUDA 設備（RTX 3060 Ti）
        torch.cuda.set_device(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("成功偵測到 Apple MPS 設備！將使用 MPS 進行訓練。")
    else:
        device = torch.device("cpu")
        print("警告：未偵測到 CUDA 設備。將使用 CPU 進行訓練。")
    return device

def compute_class_weights(dataset: ImageFolder) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    targets = np.array([y for _, y in dataset.samples])
    class_counts = np.bincount(targets, minlength=len(dataset.classes)).astype(np.float32)
    class_weights = (1.0 / np.maximum(class_counts, 1.0))
    class_weights *= (len(dataset.classes) / class_weights.sum())
    return torch.tensor(class_weights, dtype=torch.float32)

def build_transforms(img_size: int = 224, use_randaugment: bool = False, rotate_deg: int = 20):
    train_tfms = [
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=rotate_deg),
    ]
    if use_randaugment:
        try:
            from torchvision.transforms import RandAugment
            train_tfms.append(RandAugment(num_ops=2, magnitude=9))
        except Exception:
            print("[WARN] RandAugment not available in this torchvision version. Skipping.")

    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transforms.Compose(train_tfms), val_tfms

#創建tiff文件夾
from PIL import Image

class CustomTiffDataset(torch.utils.data.Dataset):
    """自訂的 Dataset，用於讀取單一資料夾中的 .tiff 檔案"""
    def __init__(self, image_paths, labels, class_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 讀取影像
        image_path = self.image_paths[idx]
        # 使用 PIL.Image 來開啟，它對 .tiff 格式支援良好
        image = Image.open(image_path).convert('RGB')
        
        # 取得對應的數字標籤
        label_name = self.labels[idx]
        label_idx = self.class_to_idx[label_name]

        # 應用資料增強/轉換
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
    
# ---- MixUp/CutMix ----
def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_mixup(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def apply_cutmix(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return x_cut, y_a, y_b, lam

def mix_criterion(criterion, pred, y_a, y_b=None, lam=1.0):
    if y_b is None:
        return criterion(pred, y_a)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ---- Grad-CAM ----
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, class_idx: Optional[int], scores: torch.Tensor) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap for a single forward pass score tensor.
        scores: logits [N, C]
        class_idx: index of class to visualize; if None, uses predicted class.
        Returns heatmaps tensor [N, H, W] in [0,1].
        """
        if class_idx is None:
            class_idx = scores.argmax(dim=1)
        one_hot = torch.zeros_like(scores)
        for i in range(scores.size(0)):
            one_hot[i, class_idx[i]] = 1.0
        self.model.zero_grad(set_to_none=True)
        scores.backward(gradient=one_hot, retain_graph=True)

        # activations: [N, K, H, W], gradients: [N, K, H, W]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [N, K, 1, 1]
        cam = (weights * self.activations).sum(dim=1)            # [N, H, W]
        cam = F.relu(cam)
        # normalize per-sample to [0,1]
        B, H, W = cam.size(0), cam.size(1), cam.size(2)
        cam = cam.view(B, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.view(B, H, W)
        return cam

def overlay_cam_on_image(img: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
    """
    img: [3, H, W] normalized (ImageNet) tensor
    cam: [H, W] in [0,1]
    Returns: overlay image [3, H, W] in [0,1] (cpu tensor)
    """
    # de-normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3,1,1)
    img_denorm = img * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)

    cam_3 = cam.unsqueeze(0).repeat(3,1,1)  # [3,H,W]
    overlay = 0.6 * img_denorm + 0.4 * cam_3
    overlay = torch.clamp(overlay, 0, 1).cpu()
    return overlay

# ---- Model builder ----
def build_model(num_classes: int, pretrained: bool = True):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# ---- Training / Eval ----
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, scheduler, device, criterion, mixup_alpha=0.0, cutmix_alpha=0.0, epoch=0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    # ✅ tqdm 包住 dataloader
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} Training", unit="batch")

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # MixUp or CutMix (prioritize CutMix if both > 0)
        if cutmix_alpha > 0:
            images, y_a, y_b, lam = apply_cutmix(images, targets, alpha=cutmix_alpha)
        else:
            images, y_a, y_b, lam = apply_mixup(images, targets, alpha=mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = mix_criterion(criterion, logits, y_a, y_b, lam)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            # For accuracy, use y_a when mixup is enabled (approximate)
            correct += (preds == y_a).sum().item()
            total += targets.size(0)
            loss_sum += loss.item() * targets.size(0)

        # ✅ tqdm 進度條上即時顯示 loss 和 acc
        pbar.set_postfix(loss=loss.item(), acc=100.*correct/max(total, 1))

    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        loss_sum += loss.item() * targets.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

def build_loaders(
    data_root: str, train_dir: str, val_dir: str, img_size: int, batch_size: int,
    use_randaugment: bool, num_workers: int, class_balance: bool
):
    train_tfms, val_tfms = build_transforms(img_size=img_size, use_randaugment=use_randaugment)
    train_ds = ImageFolder(os.path.join(data_root, train_dir), transform=train_tfms)
    val_ds = ImageFolder(os.path.join(data_root, val_dir), transform=val_tfms)

    if class_balance:
        targets = [y for _, y in train_ds.samples]
        class_counts = np.bincount(targets, minlength=len(train_ds.classes)).astype(np.float32)
        class_weights = 1.0 / np.maximum(class_counts, 1.0)
        sample_weights = [class_weights[y] for y in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_ds, val_ds, train_loader, val_loader

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                    help="Dataset root with class subfolders")
    parser.add_argument("--val_split", type=float, default=0.2,
                    help="Validation split ratio (default=0.2)")
    parser.add_argument("--num_classes", type=int, default=15,
                    help="Number of WBC classes (default=15)")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--epochs_head", type=int, default=8)
    parser.add_argument("--epochs_ft", type=int, default=20)

    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)  # for SGD

    parser.add_argument("--use_randaugment", action="store_true")
    parser.add_argument("--rotate_deg", type=int, default=20)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=0.0)

    parser.add_argument("--class_balance", action="store_true", help="Use WeightedRandomSampler for train loader")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights in CE loss")

    parser.add_argument("--save_dir", type=str, default="/Users/wdwddaniel/Desktop/side_projects/ALL_cells/model")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gradcam_samples", type=int, default=0, help="Save N Grad-CAM overlays from validation set")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    # Data
    '''
    train_ds, val_ds, train_loader, val_loader = build_loaders(
        args.data_root, args.train_dir, args.val_dir, args.img_size, args.batch_size,
        args.use_randaugment, args.num_workers, args.class_balance
    )
    num_classes = args.num_classes
    assert num_classes == len(train_ds.classes), "num_classes must match train dataset classes."
    '''
    # 1. 手動掃描您的 .tiff 資料夾，解析出所有檔案路徑和標籤
    print("[INFO] Scanning image files and parsing labels...")
    all_image_paths = []
    all_labels = []
    
    # 假設您的 TIFF 檔都放在 args.data_root 裡面
    # 您需要確保您的 .tiff 檔名格式為 '標籤_編號.tiff'
    tiff_files = list(Path('/Users/wdwddaniel/Desktop/side_projects/ALL_cells/img/data').glob('*.tiff'))
    
    for f_path in tiff_files:
        try:
            label = f_path.name.split('_')[0]
            all_image_paths.append(str(f_path))
            all_labels.append(label)
        except IndexError:
            print(f"[WARN] Skipping file with incorrect name format: {f_path.name}")
            
    # 建立類別名稱到數字索引的映射字典
    unique_classes = sorted(list(set(all_labels)))
    class_to_idx = {name: i for i, name in enumerate(unique_classes)}
    
    assert args.num_classes == len(unique_classes), \
        f"num_classes ({args.num_classes}) must match classes found in data ({len(unique_classes)})."

    # 2. 使用 train_test_split 來切分檔案路徑列表
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths, all_labels, 
        test_size=0.2, # 您可以調整驗證集的比例
        random_state=args.seed, 
        stratify=all_labels # 確保訓練和驗證集的類別比例一致
    )
    print(f"[INFO] Total images: {len(all_image_paths)}, Training: {len(train_paths)}, Validation: {len(val_paths)}")

    # 3. 建立 transforms
    train_tfms, val_tfms = build_transforms(img_size=args.img_size, use_randaugment=args.use_randaugment)
    
    # 4. 建立我們的自訂 Dataset
    train_ds = CustomTiffDataset(train_paths, train_labels, class_to_idx, transform=train_tfms)
    val_ds = CustomTiffDataset(val_paths, val_labels, class_to_idx, transform=val_tfms)

    # 5. 建立 DataLoaders (這部分和原腳本的 class_balance 邏輯類似)
    if args.class_balance:
        # ... (此處的 class_balance 邏輯可以保留，但需要對 train_ds 進行調整) ...
        # 為了簡化，我們先用 shuffle=True
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    num_classes = args.num_classes
    # Model
    model = build_model(num_classes=num_classes, pretrained=True).to(device)

    # Loss
    if args.use_class_weights:
        class_weights = compute_class_weights(train_ds).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[INFO] Using class weights: {class_weights.detach().cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer & schedulers
    # Phase 1: train head only
    for p in model.features.parameters():
        p.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(head_params, lr=args.lr_head, momentum=args.momentum, weight_decay=args.weight_decay)

    # OneCycleLR needs total steps
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr_head, epochs=args.epochs_head, steps_per_epoch=steps_per_epoch
    )

    # Train head
    best_val_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(1, args.epochs_head + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion,
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha
        )
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0
        print(f"[HEAD] Epoch {epoch}/{args.epochs_head} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, os.path.join(args.save_dir, "best_head.pt"))

    # Phase 2: fine-tune all
    for p in model.features.parameters():
        p.requires_grad = True

    # Differential learning rates
    params = [
        {"params": model.features.parameters(), "lr": args.lr_backbone},
        {"params": model.classifier.parameters(), "lr": args.lr_head},
    ]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr_backbone, args.lr_head], epochs=args.epochs_ft, steps_per_epoch=steps_per_epoch
    )

    for epoch in range(1, args.epochs_ft + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion,
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha
        )
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0
        print(f"[FT  ] Epoch {epoch}/{args.epochs_ft} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, args.epochs_head + epoch, os.path.join(args.save_dir, "best_finetune.pt"))

    # Grad-CAM sanity check (optional)
    if args.gradcam_samples > 0:
        # pick last Conv2d in features
        target_layer = None
        for m in reversed(model.features):
            if isinstance(m, nn.Conv2d):
                target_layer = m
                break
            # EfficientNet features are nested; search recursively
            if hasattr(m, "children"):
                for sub in reversed(list(m.modules())):
                    if isinstance(sub, nn.Conv2d):
                        target_layer = sub
                        break
            if target_layer is not None:
                break

        if target_layer is None:
            print("[WARN] Could not find Conv2d layer for Grad-CAM.")
        else:
            print(f"[INFO] Using Grad-CAM target layer: {target_layer}")
            gradcam = GradCAM(model, target_layer)

            # Take a few validation samples
            val_loader_cam = DataLoader(val_ds, batch_size=1, shuffle=True)
            save_cam_dir = os.path.join(args.save_dir, "gradcam")
            os.makedirs(save_cam_dir, exist_ok=True)
            model.eval()
            count = 0
            for img, label in val_loader_cam:
                img = img.to(device)
                logits = model(img)
                cams = gradcam(class_idx=None, scores=logits)  # predicted class
                overlay = overlay_cam_on_image(img[0], cams[0])  # [3,H,W]
                # save
                import torchvision.utils as vutils
                out_path = os.path.join(save_cam_dir, f"cam_{count:03d}.png")
                vutils.save_image(overlay, out_path)
                count += 1
                if count >= args.gradcam_samples:
                    break
            gradcam.remove_hooks()
            print(f"[INFO] Saved {count} Grad-CAM overlays to {save_cam_dir}")

if __name__ == "__main__":
    main()
