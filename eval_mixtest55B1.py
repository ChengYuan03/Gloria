# ============================================================
# 混合测试集评估脚本（5:5）
#
# 直接加载增强组1已训练好的最佳模型，
# 只把测试集换成混合测试集，不需要重新训练
#
# 相比 eval_only_syn3k.py 的修改点（共2处）：
#
#   [修改1] 测试集构建：真实622张取一半 + 合成数据取相同数量（5:5）
#           eval_only_syn3k: test_df = 622张纯真实
#           本文件:          test_df = 311张真实 + 311张合成
#
#   [修改2] experiment 字段及打印标题
#           eval_only_syn3k: 'real3k_syn3k'
#           本文件:          'real3k_syn3k_mixtest55'
#
# OUTPUT_DIR 不变，结果保存在增强组1的目录下（新建子文件夹）
# ============================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, classification_report,
    roc_curve, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from PIL import Image

# ── 路径配置 ──────────────────────────────────────────────
DATA_ROOT  = r"D:\BN5207_final\BN5207_data_real_images\stage0_training_handoff"
IMAGE_ROOT = r"D:\BN5207_final\BN5207_data_real_images\stage0_training_handoff"

# 加载增强组1训练好的最佳模型（不变）
OUTPUT_DIR = r"C:\Users\Reggie AugardSenft\gloria\output\ExpB_real3k_syn3k"

# [修改1] 合成数据路径，用于构建混合测试集
SYN_CSV  = r"D:\BN5207_final\BN5207_data_syn_images_comple 2.0\synthetic.csv"
SEED     = 42

# 结果单独保存到子文件夹，不覆盖原来的结果
MIXTEST_OUTPUT_DIR = r"C:\Users\Reggie AugardSenft\gloria\output\ExpB_real3k_syn3k\mixtest55"

BATCH_SIZE  = 32
NUM_WORKERS = 0
NUM_CLASSES = 5
LABEL_NAMES = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Pleural Other', 'Pleural Effusion']
LABEL_MAP   = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_root, transform=None):
        self.df         = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row          = self.df.iloc[idx]
        img_rel_path = str(row['P_real'])

        # 测试集全是真实数据，P_real 是相对路径，直接拼接 IMAGE_ROOT
        if os.path.isabs(img_rel_path):
            img_path = img_rel_path
        else:
            img_path = os.path.join(self.image_root, img_rel_path)

        img_path = img_path.replace('/', os.sep).replace('\\', os.sep)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), 0)

        if self.transform:
            image = self.transform(image)

        return image, LABEL_MAP[int(row['label'])]


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 加载模型 ───────────────────────────────────────────
    from gloria_exp_syn3k import GLoRIAClassifier

    # 训练在解冻后停止，所以 freeze_encoder=False
    model = GLoRIAClassifier(
        num_classes=NUM_CLASSES,
        freeze_encoder=False
    ).to(device)

    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型文件: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✅ 加载最佳模型，来自 Epoch {ckpt['epoch']}")
    print(f"   最佳验证集 AUC: {ckpt['best_val_auc']:.4f}")

    # ── 构建混合测试集（5:5）[修改1] ──────────────────────
    import random
    random.seed(SEED)

    real_test_df = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
    syn_df       = pd.read_csv(SYN_CSV)

    # 真实测试集取一半
    real_test_half = real_test_df.sample(frac=0.5, random_state=SEED)

    # 合成数据随机取相同数量（允许和训练集有重叠，因为无法区分）
    syn_test_half = syn_df.sample(n=len(real_test_half), random_state=SEED + 1)

    # 合并成混合测试集
    test_df = pd.concat([real_test_half, syn_test_half], ignore_index=True)

    print(f"\n混合测试集构成（5:5）：")
    print(f"  真实部分: {len(real_test_half)} 张")
    print(f"  合成部分: {len(syn_test_half)} 张")
    print(f"  合计:     {len(test_df)} 张")

    test_dataset = CXRDataset(test_df, IMAGE_ROOT, test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    print(f"\n测试集: {len(test_df)} 张（纯真实数据 ✅）")

    # ── 推理 ───────────────────────────────────────────────
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            probs          = torch.softmax(outputs, dim=1)
            _, predicted   = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_labels = np.array(all_labels)
    test_preds  = np.array(all_preds)
    test_probs  = np.array(all_probs)
    labels_bin  = label_binarize(test_labels, classes=list(range(NUM_CLASSES)))

    # ── 总体指标 ───────────────────────────────────────────
    auc       = roc_auc_score(labels_bin, test_probs,
                               average='macro', multi_class='ovr')
    accuracy  = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds,
                                average='macro', zero_division=0)
    recall    = recall_score(test_labels, test_preds,
                             average='macro', zero_division=0)
    f1        = f1_score(test_labels, test_preds,
                         average='macro', zero_division=0)

    # [修改2] 打印标题
    print(f"\n{'='*50}")
    print("📊 混合测试集评估结果（增强组1模型 / 测试集5:5）")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"{'='*50}")

    # ── 各病种单独 AUC ─────────────────────────────────────
    print("\n📊 各病种单独 AUC：")
    per_class_auc = {}
    for i, name in enumerate(LABEL_NAMES):
        try:
            auc_i = roc_auc_score(labels_bin[:, i], test_probs[:, i])
            per_class_auc[name] = round(auc_i, 4)
            print(f"  {name:<20}: {auc_i:.4f}")
        except Exception as e:
            per_class_auc[name] = None
            print(f"  {name:<20}: N/A")

    # ── 各病种详细报告 ─────────────────────────────────────
    print("\n📋 各病种详细报告：")
    print(classification_report(
        test_labels, test_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=LABEL_NAMES,
        digits=4, zero_division=0))

    # ── 保存结果 JSON ──────────────────────────────────────
    p_per = precision_score(test_labels, test_preds,
                            labels=list(range(NUM_CLASSES)),
                            average=None, zero_division=0)
    r_per = recall_score(test_labels, test_preds,
                         labels=list(range(NUM_CLASSES)),
                         average=None, zero_division=0)
    f_per = f1_score(test_labels, test_preds,
                     labels=list(range(NUM_CLASSES)),
                     average=None, zero_division=0)

    final_results = {
        'experiment':      'real3k_syn3k_mixtest55',   # [修改2]
        'best_epoch':      ckpt.get('epoch'),
        'best_val_auc':    ckpt.get('best_val_auc'),
        'test_size':       len(test_df),
        'test_real_size':  len(real_test_half),
        'test_syn_size':   len(syn_test_half),
        'overall': {
            'accuracy':  round(float(accuracy),  4),
            'auc':       round(float(auc),        4),
            'precision': round(float(precision),  4),
            'recall':    round(float(recall),     4),
            'f1':        round(float(f1),         4),
        },
        'per_class_auc': per_class_auc,
        'per_class_prf': {
            name: {
                'precision': round(float(p_per[i]), 4),
                'recall':    round(float(r_per[i]), 4),
                'f1':        round(float(f_per[i]), 4),
            }
            for i, name in enumerate(LABEL_NAMES)
        }
    }

    os.makedirs(MIXTEST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(MIXTEST_OUTPUT_DIR, "plots"), exist_ok=True)
    result_path = os.path.join(MIXTEST_OUTPUT_DIR, "final_results_mixtest55.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存到: {result_path}")

    # ── ROC 曲线 ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['steelblue', 'tomato', 'green', 'purple', 'orange']
    for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
        try:
            fpr, tpr, _ = roc_curve(labels_bin[:, i], test_probs[:, i])
            auc_i       = roc_auc_score(labels_bin[:, i], test_probs[:, i])
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{name} (AUC={auc_i:.4f})')
        except:
            pass
    ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Exp B: Real 3k + Synthetic 3k [Test Set]')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MIXTEST_OUTPUT_DIR, "plots", "roc_curves_mixtest55.png"), dpi=150)
    plt.close()
    print("✅ ROC 曲线已保存")

    # ── 混淆矩阵 ───────────────────────────────────────────
    cm = confusion_matrix(test_labels, test_preds,
                          labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix - Exp B: Real 3k + Synthetic 3k [Test Set]')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(MIXTEST_OUTPUT_DIR, "plots", "confusion_matrix_mixtest55.png"), dpi=150)
    plt.close()
    print("✅ 混淆矩阵已保存")

    print(f"\n🎉 所有结果保存在: {MIXTEST_OUTPUT_DIR}")
