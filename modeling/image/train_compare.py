"""
Step 4+5: 멀티헤드 분류 모델 3종 학습 및 5-fold CV 성능 비교
모델: EfficientNetV2-S / ConvNeXt-Tiny / ResNet50
헤드: emotion / visual / space (3축 멀티헤드)
평가: emotion F1, visual F1, space F1, exact match, confusion matrix
"""

import os

import random
import warnings
import numpy as np
import pandas as pd
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import torchvision.models as tvm
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
CSV_PATH   = "/home/piai/다운로드/labeled_293.csv"
IMAGE_ROOT = "/home/piai/다운로드/labeled_images"
SAVE_DIR   = "/home/piai/다운로드/saved_models_compare"
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_COL   = "filename"
EMOTION_COL = "emotion"
VISUAL_COL  = "visual"
SPACE_COL   = "space"

# ── 하이퍼파라미터 ──────────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 16
EPOCHS       = 20
LR           = 3e-4
WEIGHT_DECAY = 1e-4
N_FOLDS      = 5
NUM_WORKERS  = 2

EMOTION_W = 1.0
VISUAL_W  = 1.0
SPACE_W   = 1.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# ── 전처리 ──────────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

valid_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ────────────────────────────────────────────────────────────────────
class SpaceImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        row   = self.df.iloc[idx]
        image = Image.open(row["full_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image":   image,
            "emotion": torch.tensor(int(row["emotion_idx"]), dtype=torch.long),
            "visual":  torch.tensor(int(row["visual_idx"]),  dtype=torch.long),
            "space":   torch.tensor(int(row["space_idx"]),   dtype=torch.long),
        }


# ── 모델 정의 (3종) ────────────────────────────────────────────────────────────
class MultiHeadClassifier(nn.Module):
    """공통 멀티헤드 래퍼: 백본 교체만으로 3종 모델을 통일된 인터페이스로 사용"""

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_dim: int,
        num_emotion: int,
        num_visual: int,
        num_space: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        hidden = 256
        self.neck = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(hidden, num_emotion)
        self.visual_head  = nn.Linear(hidden, num_visual)
        self.space_head   = nn.Linear(hidden, num_space)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.neck(feat)
        return self.emotion_head(feat), self.visual_head(feat), self.space_head(feat)


def build_efficientnet_v2s(num_emotion, num_visual, num_space) -> MultiHeadClassifier:
    weights  = tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    backbone = tvm.efficientnet_v2_s(weights=weights)
    out_dim  = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    return MultiHeadClassifier(backbone, out_dim, num_emotion, num_visual, num_space)


def build_convnext_tiny(num_emotion, num_visual, num_space) -> MultiHeadClassifier:
    weights  = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    backbone = tvm.convnext_tiny(weights=weights)
    out_dim  = backbone.classifier[2].in_features
    backbone.classifier = nn.Identity()
    return MultiHeadClassifier(backbone, out_dim, num_emotion, num_visual, num_space)


def build_resnet50(num_emotion, num_visual, num_space) -> MultiHeadClassifier:
    weights  = tvm.ResNet50_Weights.IMAGENET1K_V2
    backbone = tvm.resnet50(weights=weights)
    out_dim  = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return MultiHeadClassifier(backbone, out_dim, num_emotion, num_visual, num_space)


MODEL_BUILDERS = {
    "EfficientNetV2-S": build_efficientnet_v2s,
    "ConvNeXt-Tiny":    build_convnext_tiny,
    "ResNet50":         build_resnet50,
}


# ── 유틸 함수 ──────────────────────────────────────────────────────────────────
def make_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    counts  = np.maximum(counts, 1)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def exact_match(emo_t, vis_t, spa_t, emo_p, vis_p, spa_p) -> float:
    emo_t, vis_t, spa_t = np.array(emo_t), np.array(vis_t), np.array(spa_t)
    emo_p, vis_p, spa_p = np.array(emo_p), np.array(vis_p), np.array(spa_p)
    return float(((emo_t == emo_p) & (vis_t == vis_p) & (spa_t == spa_p)).mean())


# ── 1 epoch 학습 / 검증 ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criteria) -> dict:
    model.train()
    total_loss = 0.0
    crit_emo, crit_vis, crit_spa = criteria

    for batch in loader:
        img = batch["image"].to(DEVICE)
        emo = batch["emotion"].to(DEVICE)
        vis = batch["visual"].to(DEVICE)
        spa = batch["space"].to(DEVICE)

        optimizer.zero_grad()
        eo, vo, so = model(img)
        loss = (
            EMOTION_W * crit_emo(eo, emo) +
            VISUAL_W  * crit_vis(vo, vis) +
            SPACE_W   * crit_spa(so, spa)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return {"loss": total_loss / len(loader)}


@torch.no_grad()
def valid_epoch(model, loader, criteria) -> dict:
    model.eval()
    total_loss = 0.0
    crit_emo, crit_vis, crit_spa = criteria
    emo_true, emo_pred = [], []
    vis_true, vis_pred = [], []
    spa_true, spa_pred = [], []

    for batch in loader:
        img = batch["image"].to(DEVICE)
        emo = batch["emotion"].to(DEVICE)
        vis = batch["visual"].to(DEVICE)
        spa = batch["space"].to(DEVICE)

        eo, vo, so = model(img)
        loss = (
            EMOTION_W * crit_emo(eo, emo) +
            VISUAL_W  * crit_vis(vo, vis) +
            SPACE_W   * crit_spa(so, spa)
        )
        total_loss += loss.item()

        emo_true.extend(emo.cpu().numpy()); emo_pred.extend(eo.argmax(1).cpu().numpy())
        vis_true.extend(vis.cpu().numpy()); vis_pred.extend(vo.argmax(1).cpu().numpy())
        spa_true.extend(spa.cpu().numpy()); spa_pred.extend(so.argmax(1).cpu().numpy())

    return {
        "loss":        total_loss / len(loader),
        "emotion_f1":  f1_score(emo_true, emo_pred, average="macro", zero_division=0),
        "visual_f1":   f1_score(vis_true, vis_pred, average="macro", zero_division=0),
        "space_f1":    f1_score(spa_true, spa_pred, average="macro", zero_division=0),
        "exact_match": exact_match(emo_true, vis_true, spa_true, emo_pred, vis_pred, spa_pred),
        "emotion_true": emo_true, "emotion_pred": emo_pred,
        "visual_true":  vis_true, "visual_pred":  vis_pred,
        "space_true":   spa_true, "space_pred":   spa_pred,
    }


# ── 5-fold CV 학습 ─────────────────────────────────────────────────────────────
def run_kfold(
    model_name: str,
    df: pd.DataFrame,
    num_emotion: int,
    num_visual: int,
    num_space: int,
) -> dict:
    build_fn = MODEL_BUILDERS[model_name]
    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # 계층 분할 키: 3축 복합 라벨
    strat_key = (
        df["emotion_idx"].astype(str) + "_" +
        df["visual_idx"].astype(str)  + "_" +
        df["space_idx"].astype(str)
    ).values

    fold_results = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(df, strat_key)):
        print(f"  [{model_name}] Fold {fold_idx + 1}/{N_FOLDS}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        # 클래스 가중치 (train fold 기준)
        crit_emo = nn.CrossEntropyLoss(weight=make_class_weights(train_df["emotion_idx"].values, num_emotion))
        crit_vis = nn.CrossEntropyLoss(weight=make_class_weights(train_df["visual_idx"].values,  num_visual))
        crit_spa = nn.CrossEntropyLoss(weight=make_class_weights(train_df["space_idx"].values,   num_space))
        criteria = (crit_emo, crit_vis, crit_spa)

        train_ds = SpaceImageDataset(train_df, train_tf)
        valid_ds = SpaceImageDataset(valid_df, valid_tf)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        model     = build_fn(num_emotion, num_visual, num_space).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_score   = -1.0
        best_metrics = None

        for epoch in range(1, EPOCHS + 1):
            train_epoch(model, train_loader, optimizer, criteria)
            val = valid_epoch(model, valid_loader, criteria)
            scheduler.step()

            score = (val["emotion_f1"] + val["visual_f1"] + val["space_f1"]) / 3
            if score > best_score:
                best_score   = score
                best_metrics = val

        fold_results.append({
            "fold":        fold_idx + 1,
            "emotion_f1":  best_metrics["emotion_f1"],
            "visual_f1":   best_metrics["visual_f1"],
            "space_f1":    best_metrics["space_f1"],
            "exact_match": best_metrics["exact_match"],
            # confusion matrix용 raw 예측 보관
            "emotion_true": best_metrics["emotion_true"],
            "emotion_pred": best_metrics["emotion_pred"],
            "visual_true":  best_metrics["visual_true"],
            "visual_pred":  best_metrics["visual_pred"],
            "space_true":   best_metrics["space_true"],
            "space_pred":   best_metrics["space_pred"],
        })

    # fold 평균
    avg = {
        "model":       model_name,
        "emotion_f1":  np.mean([r["emotion_f1"]  for r in fold_results]),
        "visual_f1":   np.mean([r["visual_f1"]   for r in fold_results]),
        "space_f1":    np.mean([r["space_f1"]    for r in fold_results]),
        "exact_match": np.mean([r["exact_match"] for r in fold_results]),
        "mean_f1":     np.mean([
            (r["emotion_f1"] + r["visual_f1"] + r["space_f1"]) / 3
            for r in fold_results
        ]),
        "emotion_f1_std":  np.std([r["emotion_f1"]  for r in fold_results]),
        "visual_f1_std":   np.std([r["visual_f1"]   for r in fold_results]),
        "space_f1_std":    np.std([r["space_f1"]    for r in fold_results]),
        "fold_results":    fold_results,
    }
    return avg


# ── confusion matrix 저장 ──────────────────────────────────────────────────────
def save_confusion_matrices(
    model_name: str,
    fold_results: list,
    emotion_classes: list,
    visual_classes: list,
    space_classes: list,
) -> None:
    # 전체 fold 예측 합산
    emo_t = sum([r["emotion_true"] for r in fold_results], [])
    emo_p = sum([r["emotion_pred"] for r in fold_results], [])
    vis_t = sum([r["visual_true"]  for r in fold_results], [])
    vis_p = sum([r["visual_pred"]  for r in fold_results], [])
    spa_t = sum([r["space_true"]   for r in fold_results], [])
    spa_p = sum([r["space_pred"]   for r in fold_results], [])

    configs = [
        ("emotion", emo_t, emo_p, emotion_classes),
        ("visual",  vis_t, vis_p, visual_classes),
        ("space",   spa_t, spa_p, space_classes),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Confusion Matrix — {model_name} (all folds)", fontsize=14)

    for ax, (axis_name, true, pred, classes) in zip(axes, configs):
        cm = confusion_matrix(true, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(axis_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"cm_{model_name.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  confusion matrix 저장: {save_path}")


# ── 비교 차트 저장 ─────────────────────────────────────────────────────────────
def save_comparison_chart(summary_df: pd.DataFrame) -> None:
    metrics = ["emotion_f1", "visual_f1", "space_f1", "exact_match", "mean_f1"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (_, row) in enumerate(summary_df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["model"])

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("5-Fold CV 모델 성능 비교")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "model_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"비교 차트 저장: {save_path}")


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Device: {DEVICE}")

    # ── 데이터 로드 & 전처리 ──────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"CSV rows: {len(df)}")

    df[EMOTION_COL] = df[EMOTION_COL].str.strip().str.lower()
    df[VISUAL_COL]  = df[VISUAL_COL].str.strip().str.lower()
    df[SPACE_COL]   = df[SPACE_COL].str.strip().str.lower()

    def build_full_path(filename):
        return os.path.join(IMAGE_ROOT, str(filename))

    df["full_path"] = df[IMAGE_COL].apply(build_full_path)
    df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"실제 존재하는 이미지: {len(df)}개")

    emotion_le = LabelEncoder()
    visual_le  = LabelEncoder()
    space_le   = LabelEncoder()

    df["emotion_idx"] = emotion_le.fit_transform(df[EMOTION_COL])
    df["visual_idx"]  = visual_le.fit_transform(df[VISUAL_COL])
    df["space_idx"]   = space_le.fit_transform(df[SPACE_COL])

    num_emotion = df["emotion_idx"].nunique()
    num_visual  = df["visual_idx"].nunique()
    num_space   = df["space_idx"].nunique()

    print(f"emotion ({num_emotion}): {list(emotion_le.classes_)}")
    print(f"visual  ({num_visual}):  {list(visual_le.classes_)}")
    print(f"space   ({num_space}):   {list(space_le.classes_)}")

    # ── 3종 모델 순차 학습 ────────────────────────────────────────────────────
    all_results = []

    for model_name in MODEL_BUILDERS:
        print(f"\n{'='*60}")
        print(f"모델: {model_name}")
        print(f"{'='*60}")

        result = run_kfold(model_name, df, num_emotion, num_visual, num_space)
        all_results.append(result)

        print(f"\n  [평균] emotion_f1={result['emotion_f1']:.4f}(±{result['emotion_f1_std']:.3f})"
              f"  visual_f1={result['visual_f1']:.4f}(±{result['visual_f1_std']:.3f})"
              f"  space_f1={result['space_f1']:.4f}(±{result['space_f1_std']:.3f})"
              f"  exact_match={result['exact_match']:.4f}"
              f"  mean_f1={result['mean_f1']:.4f}")

        save_confusion_matrices(
            model_name,
            result["fold_results"],
            list(emotion_le.classes_),
            list(visual_le.classes_),
            list(space_le.classes_),
        )

    # ── 비교 요약표 ───────────────────────────────────────────────────────────
    summary_cols = ["model", "emotion_f1", "visual_f1", "space_f1",
                    "exact_match", "mean_f1",
                    "emotion_f1_std", "visual_f1_std", "space_f1_std"]
    summary_df   = pd.DataFrame([{c: r[c] for c in summary_cols} for r in all_results])

    print("\n" + "="*60)
    print("최종 비교 결과 (5-fold 평균)")
    print("="*60)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary_df.to_string(index=False))

    best_row = summary_df.loc[summary_df["mean_f1"].idxmax()]
    print(f"\n★ 최고 성능 모델: {best_row['model']}  (mean_f1={best_row['mean_f1']:.4f})")

    # CSV 저장
    csv_path = os.path.join(SAVE_DIR, "comparison_results.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"결과 CSV 저장: {csv_path}")

    save_comparison_chart(summary_df)

    # ── 축별 classification report (전체 fold 합산) ───────────────────────────
    print("\n── 최고 모델 축별 리포트 ──")
    best_result = next(r for r in all_results if r["model"] == best_row["model"])
    fold_results = best_result["fold_results"]

    for axis, true_key, pred_key, classes in [
        ("emotion", "emotion_true", "emotion_pred", emotion_le.classes_),
        ("visual",  "visual_true",  "visual_pred",  visual_le.classes_),
        ("space",   "space_true",   "space_pred",   space_le.classes_),
    ]:
        all_true = sum([r[true_key] for r in fold_results], [])
        all_pred = sum([r[pred_key] for r in fold_results], [])
        print(f"\n=== {axis} ===")
        print(classification_report(all_true, all_pred, target_names=classes, zero_division=0))


if __name__ == "__main__":
    main()
