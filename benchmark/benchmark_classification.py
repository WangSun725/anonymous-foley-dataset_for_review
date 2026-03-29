import os
os.environ["TORCHVISION_DISABLE_TORCHCODEC"] = "1"

import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

from hear21passt.base import load_model, get_scene_embeddings


# =========================================================
# 0. Config
# =========================================================
DATA_ROOT = "/root/autodl-tmp/FoleySet"
OUTPUT_DIR = "/root/autodl-tmp/passt_subcategory_runs" 

TRAIN_CSV = os.path.join(DATA_ROOT, "train", "train.csv")
VAL_CSV   = os.path.join(DATA_ROOT, "val", "val.csv")
TEST_CSV  = os.path.join(DATA_ROOT, "test", "test.csv")

NAME_COL = "name"
LABEL_COL = "sub-category"  

SAMPLE_RATE = 32000
CLIP_SECONDS = 5.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

CACHE_EMB = True
EMB_CACHE_DIR = os.path.join(OUTPUT_DIR, "emb_cache")

CACHE_BATCH_SIZE = 32
CACHE_NUM_WORKERS = 0

HEAD_BATCH_SIZE = 256
HEAD_NUM_WORKERS = 0

EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4

FREEZE_BACKBONE = True
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 1. Utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def resolve_audio_path(split_root: Path, name_value: str) -> Path:
    name_value = str(name_value)
    if name_value.lower().endswith(".wav"):
        return split_root / name_value
    return split_root / f"{name_value}.wav"

def load_audio_mono_32k(path: Path, target_sr: int = 32000) -> torch.Tensor:
    """
    用 soundfile 读 wav，避免 torchaudio.load -> torchcodec
    返回 torch Tensor [T] float32
    """
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)

    if x.ndim == 2:
        x = x.mean(axis=1)  # to mono

    if sr != target_sr:
        x_t = torch.from_numpy(x).unsqueeze(0)  # [1, T]
        x_t = torchaudio.functional.resample(x_t, sr, target_sr)
        x = x_t.squeeze(0).numpy()

    return torch.from_numpy(x)

def pad_to_len(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = wav.shape[0]
    if cur == target_len:
        return wav
    if cur > target_len:
        return wav[:target_len]
    out = torch.zeros(target_len, dtype=wav.dtype)
    out[:cur] = wav
    return out


# =========================================================
# 2. Audio Dataset (for caching embeddings)
# =========================================================
class CategoryAudioDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split_dir: str,
        label2id: dict,
        name_col: str = "name",
        label_col: str = "sub-category",
        clip_samples: int = 160000,
    ):
        self.df = pd.read_csv(csv_path).copy()
        self.split_dir = Path(split_dir)
        self.label2id = label2id
        self.name_col = name_col
        self.label_col = label_col
        self.clip_samples = clip_samples

        missing_cols = [c for c in [name_col, label_col] if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"{csv_path} 缺少列: {missing_cols}")

        self.df[label_col] = self.df[label_col].astype(str).str.strip()
        self.df = self.df[self.df[label_col].isin(label2id.keys())].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = resolve_audio_path(self.split_dir, row[self.name_col])
        if not audio_path.exists():
            raise FileNotFoundError(f"找不到音频文件: {audio_path}")

        wav = load_audio_mono_32k(audio_path, SAMPLE_RATE)
        wav = pad_to_len(wav, self.clip_samples)

        label_name = str(row[self.label_col]).strip()
        label_id = self.label2id[label_name]

        return {
            "name": Path(str(audio_path)).stem,
            "waveform": wav,
            "label": torch.tensor(label_id).long(),
            "path": str(audio_path),
        }

def audio_collate_fn(batch):
    wavs = torch.stack([b["waveform"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    names = [b["name"] for b in batch]
    paths = [b["path"] for b in batch]
    return wavs, labels, names, paths


# =========================================================
# 3. PaSST Embedding Extractor
# =========================================================
class PaSSTEmbeddingExtractor(nn.Module):
    def __init__(self, device: str = "cuda", freeze_backbone: bool = True):
        super().__init__()
        self.device = device
        self.freeze_backbone = freeze_backbone

        self.backbone = load_model().to(device).eval()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, CLIP_SAMPLES, device=device)
            emb = get_scene_embeddings(dummy, self.backbone)
            self.embedding_dim = int(emb.shape[-1])

    @torch.no_grad()
    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        wavs = wavs.to(self.device, non_blocking=True)
        self.backbone.eval()
        emb = get_scene_embeddings(wavs, self.backbone)
        return emb


@torch.no_grad()
def cache_split_embeddings(
    split_name: str,
    audio_ds: Dataset,
    extractor: PaSSTEmbeddingExtractor,
    cache_root: str,
    device: str,
    batch_size: int = 32,
    num_workers: int = 0,
):
    split_dir = os.path.join(cache_root, split_name)
    ensure_dir(split_dir)

    loader = DataLoader(
        audio_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=audio_collate_fn,
    )

    total = 0
    written = 0

    for wavs, labels, names, _paths in tqdm(loader, desc=f"Caching {split_name}", leave=True):
        embs = extractor(wavs).detach().cpu()
        labels = labels.cpu().tolist()

        for i, name in enumerate(names):
            out_path = os.path.join(split_dir, f"{name}.pt")
            total += 1
            if os.path.exists(out_path):
                continue
            torch.save({"emb": embs[i], "label": int(labels[i])}, out_path)
            written += 1

    print(f"[{split_name}] cached {written}/{total} new embeddings -> {split_dir}")


# =========================================================
# 4. Embedding Dataset
# =========================================================
class EmbeddingDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        cache_split_dir: str,
        label2id: dict,
        name_col: str = "name",
        label_col: str = "sub-category",
    ):
        self.df = pd.read_csv(csv_path).copy()
        self.cache_split_dir = cache_split_dir
        self.label2id = label2id
        self.name_col = name_col
        self.label_col = label_col

        missing_cols = [c for c in [name_col, label_col] if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"{csv_path} 缺少列: {missing_cols}")

        self.df[label_col] = self.df[label_col].astype(str).str.strip()
        self.df = self.df[self.df[label_col].isin(label2id.keys())].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = str(row[self.name_col])
        if name.lower().endswith(".wav"):
            name = name[:-4]

        emb_path = os.path.join(self.cache_split_dir, f"{name}.pt")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"找不到 embedding 缓存文件: {emb_path}")

        obj = torch.load(emb_path, map_location="cpu")
        emb = obj["emb"].float()

        label_name = str(row[self.label_col]).strip()
        label_id = self.label2id[label_name]
        return emb, torch.tensor(label_id).long()

def emb_collate_fn(batch):
    embs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    return embs, labels


# =========================================================
# 5. Train / Eval
# =========================================================
def train_one_epoch_on_emb(head, loader, optimizer, criterion, device):
    head.train()
    all_loss = []
    for embs, labels in tqdm(loader, desc="Train(head)", leave=False):
        embs = embs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = head(embs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())

    return float(sum(all_loss) / max(len(all_loss), 1))

@torch.no_grad()
def evaluate_on_emb(head, loader, criterion, device, id2label):
    head.eval()
    all_loss = []
    all_preds = []
    all_targets = []

    for embs, labels in tqdm(loader, desc="Eval(head)", leave=False):
        embs = embs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = head(embs)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        all_loss.append(loss.item())
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    avg_loss = float(sum(all_loss) / max(len(all_loss), 1))
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")

    report = classification_report(
        all_targets,
        all_preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4,
        zero_division=0,
    )

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
    }

@torch.no_grad()
def get_preds_targets_on_emb(head, loader, device):
    head.eval()
    all_preds, all_targets = [], []

    for embs, labels in tqdm(loader, desc="Predict", leave=False):
        embs = embs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = head(embs)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    return all_targets, all_preds

def save_confusion_matrix(cm, labels, out_png, out_csv=None, normalize=None):
    import numpy as np

    cm_to_plot = cm.astype(float)
    if normalize == "true":
        cm_to_plot = cm_to_plot / np.clip(cm_to_plot.sum(axis=1, keepdims=True), 1e-12, None)
    elif normalize == "pred":
        cm_to_plot = cm_to_plot / np.clip(cm_to_plot.sum(axis=0, keepdims=True), 1e-12, None)
    elif normalize == "all":
        cm_to_plot = cm_to_plot / np.clip(cm_to_plot.sum(), 1e-12, None)

    plt.figure(figsize=(18, 16))   # <-- 73 类建议放大
    plt.imshow(cm_to_plot, interpolation="nearest")
    plt.title("Confusion Matrix" + (f" (norm={normalize})" if normalize else ""))
    plt.colorbar()

    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, fontsize=6)
    plt.yticks(tick_marks, labels, fontsize=6)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

    if out_csv:
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_csv)


# =========================================================
# 6. Main
# =========================================================
def main():
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(EMB_CACHE_DIR)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for split_name, dfx in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if NAME_COL not in dfx.columns:
            raise ValueError(f"{split_name}.csv 缺少列 {NAME_COL}")
        if LABEL_COL not in dfx.columns:
            raise ValueError(f"{split_name}.csv 缺少标签列 {LABEL_COL}")
        dfx[LABEL_COL] = dfx[LABEL_COL].astype(str).str.strip()

    classes = sorted(train_df[LABEL_COL].unique().tolist())
    print(f"Detected {len(classes)} classes for {LABEL_COL}.")

    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    for split_name, dfx in [("val", val_df), ("test", test_df)]:
        unseen = sorted(set(dfx[LABEL_COL].unique()) - set(classes))
        if unseen:
            raise ValueError(f"{split_name} found train u: {unseen}")

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    train_audio_ds = CategoryAudioDataset(
        TRAIN_CSV, os.path.join(DATA_ROOT, "train"), label2id,
        name_col=NAME_COL, label_col=LABEL_COL, clip_samples=CLIP_SAMPLES
    )
    val_audio_ds = CategoryAudioDataset(
        VAL_CSV, os.path.join(DATA_ROOT, "val"), label2id,
        name_col=NAME_COL, label_col=LABEL_COL, clip_samples=CLIP_SAMPLES
    )
    test_audio_ds = CategoryAudioDataset(
        TEST_CSV, os.path.join(DATA_ROOT, "test"), label2id,
        name_col=NAME_COL, label_col=LABEL_COL, clip_samples=CLIP_SAMPLES
    )

    extractor = PaSSTEmbeddingExtractor(device=DEVICE, freeze_backbone=FREEZE_BACKBONE)
    print("Device:", DEVICE)
    print("Embedding dim:", extractor.embedding_dim)
    print("Num classes:", len(label2id))

    if CACHE_EMB:
        cache_split_embeddings(
            "train", train_audio_ds, extractor, EMB_CACHE_DIR, DEVICE,
            batch_size=CACHE_BATCH_SIZE, num_workers=CACHE_NUM_WORKERS
        )
        cache_split_embeddings(
            "val", val_audio_ds, extractor, EMB_CACHE_DIR, DEVICE,
            batch_size=CACHE_BATCH_SIZE, num_workers=CACHE_NUM_WORKERS
        )
        cache_split_embeddings(
            "test", test_audio_ds, extractor, EMB_CACHE_DIR, DEVICE,
            batch_size=CACHE_BATCH_SIZE, num_workers=CACHE_NUM_WORKERS
        )

    train_emb_ds = EmbeddingDataset(
        TRAIN_CSV, os.path.join(EMB_CACHE_DIR, "train"), label2id,
        name_col=NAME_COL, label_col=LABEL_COL
    )
    val_emb_ds = EmbeddingDataset(
        VAL_CSV, os.path.join(EMB_CACHE_DIR, "val"), label2id,
        name_col=NAME_COL, label_col=LABEL_COL
    )
    test_emb_ds = EmbeddingDataset(
        TEST_CSV, os.path.join(EMB_CACHE_DIR, "test"), label2id,
        name_col=NAME_COL, label_col=LABEL_COL
    )

    train_loader = DataLoader(
        train_emb_ds,
        batch_size=HEAD_BATCH_SIZE,
        shuffle=True,
        num_workers=HEAD_NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=emb_collate_fn
    )
    val_loader = DataLoader(
        val_emb_ds,
        batch_size=HEAD_BATCH_SIZE,
        shuffle=False,
        num_workers=HEAD_NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=emb_collate_fn
    )
    test_loader = DataLoader(
        test_emb_ds,
        batch_size=HEAD_BATCH_SIZE,
        shuffle=False,
        num_workers=HEAD_NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=emb_collate_fn
    )

    head = nn.Linear(extractor.embedding_dim, len(label2id)).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    counts = train_df[LABEL_COL].value_counts()
    class_counts = torch.tensor(
        [counts.get(id2label[i], 0) for i in range(len(id2label))],
        dtype=torch.float
    )
    N = class_counts.sum()
    K = len(class_counts)
    class_weights = N / (K * class_counts.clamp(min=1.0))
    class_weights = class_weights.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Class weights (first 10):", class_weights[:10].detach().cpu().numpy())

    best_val_macro_f1 = -1.0
    best_ckpt_path = os.path.join(OUTPUT_DIR, "best_head.pt")
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} (Head Only) =====")
        train_loss = train_one_epoch_on_emb(head, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate_on_emb(head, val_loader, criterion, DEVICE, id2label)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)

        print(row)
        print("\n[Val classification report]")
        print(val_metrics["report"])

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "embedding_dim": extractor.embedding_dim,
                    "config": {
                        "sample_rate": SAMPLE_RATE,
                        "clip_seconds": CLIP_SECONDS,
                        "label_col": LABEL_COL,
                        "name_col": NAME_COL,
                    },
                },
                best_ckpt_path
            )
            print(f"Saved best head checkpoint to {best_ckpt_path}")

        pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "train_log.csv"), index=False)

    print("\n===== Final Test (Best Head) =====")
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
    head.load_state_dict(ckpt["head_state_dict"])

    test_metrics = evaluate_on_emb(head, test_loader, criterion, DEVICE, id2label)

    y_true, y_pred = get_preds_targets_on_emb(head, test_loader, DEVICE)
    labels_in_order = [id2label[i] for i in range(len(id2label))]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_in_order))))

    save_confusion_matrix(
        cm,
        labels_in_order,
        out_png=os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"),
        out_csv=os.path.join(OUTPUT_DIR, "confusion_matrix_test.csv"),
        normalize=None
    )

    save_confusion_matrix(
        cm,
        labels_in_order,
        out_png=os.path.join(OUTPUT_DIR, "confusion_matrix_test_norm_true.png"),
        out_csv=None,
        normalize="true"
    )

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "true_label": [labels_in_order[i] for i in y_true],
        "pred_label": [labels_in_order[i] for i in y_pred],
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    print("Saved confusion matrix + predictions to", OUTPUT_DIR)
    print({
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
    })
    print("\n[Test classification report]")
    print(test_metrics["report"])

    with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_metrics["report"])


if __name__ == "__main__":
    main()

