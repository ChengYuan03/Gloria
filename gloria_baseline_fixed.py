# ============================================================
# GLoRIA 基线实验 - 真实数据训练脚本（最终修正版）
#
# 主要修复 / 优化：
#   1. 训练/验证/测试严格分离：只用 val 选模型，test 只在最后评估一次
#   2. 冻结 encoder 时强制保持 eval 模式，避免 BN/Dropout 在训练中继续变化
#   3. feature_dim 动态推断时临时切到 eval 模式，避免不稳定
#   4. best_model.pth 保证至少首轮能保存，避免训练结束后加载失败
#   5. 加入 map_location，CPU/GPU 切换更稳
#   6. per-class AUC / ROC 在某类缺失时不报错
#   7. confusion_matrix 固定 labels，始终输出 5x5
#   8. label 读取更稳，避免 CSV 中类型异常导致 KeyError
#   9. baseline 数据增强更保守，减少医学图像不合理增强
#  10. 设置随机种子，提升可复现性
# ============================================================

import os
import sys
import json
import time
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, classification_report,
    roc_curve, confusion_matrix
)
from sklearn.preprocessing import label_binarize

import gloria


# ============================================================
# 路径配置
# ============================================================

DATA_ROOT  = r"D:\BN5207_final\BN5207_data_real_images\stage0_training_handoff"
IMAGE_ROOT = r"D:\BN5207_final\BN5207_data_real_images\stage0_training_handoff"

# 注意：
# 当前 gloria.load_gloria(device=...) 是否使用这个路径，取决于你本地 gloria 包的实现。
# 如果 load_gloria 不读取这个变量，那它只是留作说明用途。
WEIGHT_PATH = r"C:\Users\Reggie AugardSenft\gloria\pretrained\chexpert_resnet50.ckpt"

OUTPUT_DIR = r"C:\Users\Reggie AugardSenft\gloria\output\BaselineA_output"


# ============================================================
# 训练配置
# ============================================================

SEED            = 42
BATCH_SIZE      = 16
NUM_EPOCHS      = 50
HEAD_LR         = 5e-4
ENCODER_LR      = 5e-5
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 0      # Windows 下建议 0
NUM_CLASSES     = 5
FREEZE_ENCODER  = True
UNFREEZE_AT_EPOCH = 5
VAL_SPLIT       = 0.15
USE_BALANCED_SAMPLER = True
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM  = 1.0
MODEL_SELECTION_METRIC = "f1"  # 可选: "f1" / "auc"

# CSV 原始标签 1-5 -> 模型输入标签 0-4
# 请确认你自己的 CSV 标签定义和这里完全一致
LABEL_NAMES = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Pleural Other', 'Pleural Effusion']
LABEL_MAP   = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


# ============================================================
# 基础设置
# ============================================================

warnings.filterwarnings("ignore", category=UserWarning)

os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================

class CXRDataset(Dataset):
    def __init__(self, df, image_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_rel_path = str(row['P_real'])
        img_path = os.path.join(self.image_root, img_rel_path)
        img_path = img_path.replace('/', os.sep).replace('\\', os.sep)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 无法读取图像: {img_path} | 错误: {e}")
            image = Image.new('RGB', (224, 224), 0)

        if self.transform is not None:
            image = self.transform(image)

        try:
            raw_label = int(row['label'])
        except Exception:
            raise ValueError(f"CSV 中 label 无法转成 int：{row['label']}")

        if raw_label not in LABEL_MAP:
            raise ValueError(f"发现未定义标签值: {raw_label}，请检查 CSV 与 LABEL_MAP 是否一致")

        label = LABEL_MAP[raw_label]
        return image, label


# baseline 尽量保守一些，避免对医学图像做过强增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================================================
# 模型
# ============================================================

class GLoRIAClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_encoder=True):
        super().__init__()

        self.encoder_frozen = freeze_encoder
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载 GLoRIA
        gloria_model = gloria.load_gloria(device=device)
        self.encoder = gloria_model.img_encoder

        # 先按 freeze_encoder 决定参数是否训练
        self.set_encoder_trainable(trainable=not freeze_encoder)

        # 动态推断 feature_dim
        feature_dim = self._infer_feature_dim(device)
        print(f"✅ 动态推断 feature_dim = {feature_dim}")

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def set_encoder_trainable(self, trainable: bool):
        self.encoder_frozen = not trainable
        for param in self.encoder.parameters():
            param.requires_grad = trainable

        if trainable:
            self.encoder.train()
            print("✅ 编码器参与微调")
        else:
            self.encoder.eval()
            print("✅ 编码器已冻结，并固定为 eval 模式")

    def _infer_feature_dim(self, device):
        dummy = torch.zeros(1, 3, 224, 224).to(device)

        was_training = self.encoder.training
        self.encoder.eval()

        with torch.no_grad():
            out = self.encoder(dummy)

        if was_training and not self.encoder_frozen:
            self.encoder.train()

        if isinstance(out, (tuple, list)):
            out = out[0]

        if out.dim() == 4:
            out = out.mean(dim=[2, 3])

        if out.dim() != 2:
            raise RuntimeError(f"encoder 输出维度异常: {out.shape}")

        return out.shape[1]

    def forward(self, x):
        if self.encoder_frozen:
            # 冻结时禁用梯度 + 保持 eval 逻辑
            self.encoder.eval()
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)

        if isinstance(features, (tuple, list)):
            features = features[0]

        if features.dim() == 4:
            features = features.mean(dim=[2, 3])

        return self.classifier(features)


# ============================================================
# 训练 / 评估函数
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    # 关键：冻结 encoder 时，分类头 train，encoder eval
    if model.encoder_frozen:
        model.encoder.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, probs):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    try:
        auc = roc_auc_score(labels_bin, probs, average='macro', multi_class='ovr')
    except Exception as e:
        print(f"⚠️ 宏平均 AUC 计算警告: {e}")
        auc = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def safe_per_class_auc(labels_bin, probs, class_names):
    results = {}
    for i, name in enumerate(class_names):
        try:
            auc_i = roc_auc_score(labels_bin[:, i], probs[:, i])
            results[name] = round(float(auc_i), 4)
        except Exception as e:
            print(f"⚠️ {name} 的 AUC 无法计算: {e}")
            results[name] = None
    return results


def build_optimizer(model):
    head_params = list(model.classifier.parameters())
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]

    if len(encoder_params) == 0:
        optimizer = optim.Adam(
            head_params,
            lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY
        )
    else:
        optimizer = optim.Adam(
            [
                {'params': encoder_params, 'lr': ENCODER_LR},
                {'params': head_params, 'lr': HEAD_LR}
            ],
            weight_decay=WEIGHT_DECAY
        )

    return optimizer


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pin = torch.cuda.is_available()

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------
    train_csv = os.path.join(DATA_ROOT, "train.csv")
    test_csv  = os.path.join(DATA_ROOT, "test.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"未找到 train.csv: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"未找到 test.csv: {test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    required_cols = ['P_real', 'label']
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"train.csv 缺少必要列: {col}")
        if col not in test_df.columns:
            raise ValueError(f"test.csv 缺少必要列: {col}")

    print(f"\n原始训练集: {len(train_df)} 张")
    print(f"测试集:     {len(test_df)} 张")

    if 'label_name' in train_df.columns:
        print("\n训练集标签分布:")
        print(train_df['label_name'].value_counts())
    else:
        print("\n训练集标签分布（label）:")
        print(train_df['label'].value_counts())

    sample = os.path.join(IMAGE_ROOT, str(train_df.iloc[0]['P_real']))
    sample = sample.replace('/', os.sep).replace('\\', os.sep)
    print(f"\n示例图片路径: {sample}")
    print(f"文件存在: {os.path.exists(sample)}")
    if not os.path.exists(sample):
        print("❌ 图片路径有误，请检查 IMAGE_ROOT 配置")
        sys.exit(1)

    # --------------------------------------------------------
    # 划分验证集
    # --------------------------------------------------------
    train_sub_df, val_df = train_test_split(
        train_df,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=train_df['label']
    )

    print(f"\n划分后训练集: {len(train_sub_df)} 张")
    print(f"验证集:       {len(val_df)} 张")
    print(f"测试集:       {len(test_df)} 张（仅最终评估使用）")

    # --------------------------------------------------------
    # 类别权重
    # --------------------------------------------------------
    mapped_train_labels = train_sub_df['label'].astype(int).map(LABEL_MAP)
    if mapped_train_labels.isnull().any():
        bad_rows = train_sub_df.loc[mapped_train_labels.isnull(), 'label'].unique().tolist()
        raise ValueError(f"训练集中发现未定义标签: {bad_rows}")

    label_counts = mapped_train_labels.value_counts().sort_index()
    total_samples = len(train_sub_df)

    raw_class_weights = torch.tensor(
        [total_samples / (NUM_CLASSES * label_counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float
    )
    class_weights = torch.sqrt(raw_class_weights)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)

    print(f"\n类别权重（sqrt inverse frequency）:")
    for i, (name, w) in enumerate(zip(LABEL_NAMES, class_weights.cpu())):
        print(f"  {name:<20}: {float(w):.4f}  (样本数: {label_counts.get(i, 0)})")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    train_dataset = CXRDataset(train_sub_df, IMAGE_ROOT, train_transform)
    val_dataset   = CXRDataset(val_df,       IMAGE_ROOT, test_transform)
    test_dataset  = CXRDataset(test_df,      IMAGE_ROOT, test_transform)

    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=pin
    )

    if USE_BALANCED_SAMPLER:
        sample_weights = mapped_train_labels.map(lambda x: 1.0 / label_counts.get(x, 1)).astype(float).values
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            **train_loader_kwargs,
            sampler=sampler,
            shuffle=False
        )
        print("✅ 训练集使用 WeightedRandomSampler 做类别均衡采样")
    else:
        train_loader = DataLoader(
            **train_loader_kwargs,
            shuffle=True
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin
    )

    print(f"\n训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")

    # --------------------------------------------------------
    # 初始化模型
    # --------------------------------------------------------
    print("\n正在加载 GLoRIA 预训练模型...")
    model = GLoRIAClassifier(
        num_classes=NUM_CLASSES,
        freeze_encoder=FREEZE_ENCODER
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量:     {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # --------------------------------------------------------
    # 优化器 / 损失函数
    # --------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': []
    }

    # 关键：确保第一轮一定能保存 best model
    best_val_auc = -1.0
    best_val_f1 = -1.0
    best_epoch = 0
    best_ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pth")

    print(f"\n开始训练，共 {NUM_EPOCHS} 个 epoch")
    print("=" * 60)

    # --------------------------------------------------------
    # 训练循环
    # --------------------------------------------------------
    for epoch in range(NUM_EPOCHS):
        start = time.time()

        if model.encoder_frozen and (not FREEZE_ENCODER or (epoch + 1) >= UNFREEZE_AT_EPOCH):
            print(f"\n🔓 Epoch {epoch+1}: 解冻编码器并开始小学习率微调")
            model.set_encoder_trainable(trainable=True)
            optimizer = build_optimizer(model)
            scheduler = CosineAnnealingLR(optimizer, T_max=(NUM_EPOCHS - epoch))

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_labels, val_preds, val_probs = evaluate(
            model, val_loader, criterion, device
        )

        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        scheduler.step()

        elapsed = time.time() - start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])

        # 第一轮一定保存；之后按配置指标保存最优模型
        metric_now = val_metrics['f1'] if MODEL_SELECTION_METRIC == "f1" else val_metrics['auc']
        metric_best = best_val_f1 if MODEL_SELECTION_METRIC == "f1" else best_val_auc
        improved = (metric_now > metric_best) or (
            abs(metric_now - metric_best) < 1e-12 and val_metrics['auc'] > best_val_auc
        )

        if epoch == 0 or improved:
            best_val_auc = val_metrics['auc']
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'best_val_f1': best_val_f1,
                'val_metrics': val_metrics,
                'config': {
                    'batch_size': BATCH_SIZE,
                    'num_epochs': NUM_EPOCHS,
                    'head_lr': HEAD_LR,
                    'encoder_lr': ENCODER_LR,
                    'weight_decay': WEIGHT_DECAY,
                    'freeze_encoder': FREEZE_ENCODER,
                    'unfreeze_at_epoch': UNFREEZE_AT_EPOCH,
                    'use_balanced_sampler': USE_BALANCED_SAMPLER,
                    'label_smoothing': LABEL_SMOOTHING,
                    'model_selection_metric': MODEL_SELECTION_METRIC,
                    'val_split': VAL_SPLIT,
                    'seed': SEED
                }
            }, best_ckpt_path)

            print(
                f"Epoch {epoch+1:3d} ⭐ 新最佳(val)! "
                f"AUC={best_val_auc:.4f} | "
                f"F1={val_metrics['f1']:.4f} | "
                f"Acc={val_metrics['accuracy']:.4f} | "
                f"TrainLoss={train_loss:.4f} | "
                f"ValLoss={val_loss:.4f} | "
                f"耗时={elapsed:.0f}s"
            )
        else:
            print(
                f"Epoch {epoch+1:3d}    "
                f"ValAUC={val_metrics['auc']:.4f} | "
                f"F1={val_metrics['f1']:.4f} | "
                f"Acc={val_metrics['accuracy']:.4f} | "
                f"TrainLoss={train_loss:.4f} | "
                f"ValLoss={val_loss:.4f} | "
                f"耗时={elapsed:.0f}s"
            )

    print("=" * 60)
    print(
        f"训练完成！最佳模型来自 Epoch {best_epoch} | "
        f"Val AUC={best_val_auc:.4f} | Val F1={best_val_f1:.4f} | "
        f"选择指标={MODEL_SELECTION_METRIC}"
    )

    # --------------------------------------------------------
    # 保存训练历史
    # --------------------------------------------------------
    history_path = os.path.join(OUTPUT_DIR, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"✅ 训练历史已保存到: {history_path}")

    # --------------------------------------------------------
    # 最终测试
    # --------------------------------------------------------
    print("\n加载最佳模型进行最终测试集评估...")

    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"未找到最佳模型 checkpoint: {best_ckpt_path}")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # 用 ckpt 中记录的 best_val_auc，避免和内存变量不一致
    best_val_auc = ckpt.get('best_val_auc', best_val_auc)
    best_val_f1 = ckpt.get('best_val_f1', best_val_f1)

    _, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    metrics = compute_metrics(test_labels, test_preds, test_probs)
    labels_bin = label_binarize(test_labels, classes=list(range(NUM_CLASSES)))

    print("\n" + "=" * 50)
    print("📊 最终评估结果（基线组 - 真实数据）")
    print("=" * 50)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  AUC       : {metrics['auc']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print("=" * 50)

    print("\n📊 各病种单独 AUC：")
    per_class_auc = safe_per_class_auc(labels_bin, test_probs, LABEL_NAMES)
    for name, auc_i in per_class_auc.items():
        if auc_i is None:
            print(f"  {name:<22}: N/A")
        else:
            print(f"  {name:<22}: {auc_i:.4f}")

    print("\n📋 各病种详细报告：")
    print(classification_report(
        test_labels,
        test_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0
    ))

    # --------------------------------------------------------
    # 保存数值结果
    # --------------------------------------------------------
    p_per = precision_score(
        test_labels, test_preds,
        labels=list(range(NUM_CLASSES)),
        average=None, zero_division=0
    )
    r_per = recall_score(
        test_labels, test_preds,
        labels=list(range(NUM_CLASSES)),
        average=None, zero_division=0
    )
    f_per = f1_score(
        test_labels, test_preds,
        labels=list(range(NUM_CLASSES)),
        average=None, zero_division=0
    )

    final_results = {
        'experiment': 'baseline_real_data',
        'train_size': len(train_sub_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'best_epoch': ckpt.get('epoch', best_epoch),
        'best_val_auc': round(float(best_val_auc), 4) if best_val_auc is not None else None,
        'best_val_f1': round(float(best_val_f1), 4) if best_val_f1 is not None else None,
        'class_weights': {
            n: round(float(w), 4) for n, w in zip(LABEL_NAMES, class_weights.cpu())
        },
        'overall': {
            'accuracy': round(float(metrics['accuracy']), 4),
            'auc': round(float(metrics['auc']), 4) if metrics['auc'] is not None else None,
            'precision': round(float(metrics['precision']), 4),
            'recall': round(float(metrics['recall']), 4),
            'f1': round(float(metrics['f1']), 4),
        },
        'per_class_auc': per_class_auc,
        'per_class_prf': {
            name: {
                'precision': round(float(p_per[i]), 4),
                'recall': round(float(r_per[i]), 4),
                'f1': round(float(f_per[i]), 4),
            }
            for i, name in enumerate(LABEL_NAMES)
        }
    }

    final_results_path = os.path.join(OUTPUT_DIR, "final_results.json")
    with open(final_results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存到: {final_results_path}")

    # --------------------------------------------------------
    # 画图
    # --------------------------------------------------------
    print("\n正在生成图表...")
    epochs = range(1, NUM_EPOCHS + 1)

    # 图1：训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', color='tomato')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['val_auc'], label='Val AUC', color='green')
    axes[1].axhline(y=best_val_auc, color='red', linestyle='--',
                    label=f'Best Val AUC={best_val_auc:.4f}')
    axes[1].set_title('Validation AUC Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['val_f1'], label='Val F1', color='purple')
    axes[2].plot(epochs, history['val_acc'], label='Val Acc', color='orange')
    axes[2].set_title('F1 & Accuracy Curve (Val)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Baseline (Real Data Only) - Training Curves')
    plt.tight_layout()

    training_curve_path = os.path.join(OUTPUT_DIR, "plots", "training_curves.png")
    plt.savefig(training_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存: {training_curve_path}")

    # 图2：ROC 曲线（测试集）
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['steelblue', 'tomato', 'green', 'purple', 'orange']
    plotted_any = False

    for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
        try:
            fpr, tpr, _ = roc_curve(labels_bin[:, i], test_probs[:, i])
            auc_i = roc_auc_score(labels_bin[:, i], test_probs[:, i])
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc_i:.4f})')
            plotted_any = True
        except Exception as e:
            print(f"⚠️ 跳过 {name} 的 ROC 曲线：{e}")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Baseline (Real Data Only) [Test Set]')
    if plotted_any:
        ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_curve_path = os.path.join(OUTPUT_DIR, "plots", "roc_curves.png")
    plt.savefig(roc_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC 曲线已保存: {roc_curve_path}")

    # 图3：混淆矩阵
    cm = confusion_matrix(
        test_labels,
        test_preds,
        labels=list(range(NUM_CLASSES))
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix - Baseline (Real Data Only) [Test Set]')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "plots", "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {cm_path}")

    print(f"\n🎉 所有结果保存在: {OUTPUT_DIR}")
