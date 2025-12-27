"""
lstm_pose_classifier.py

LSTM-based sequence classification (PyTorch)
- Loads NPZ files from data/poses_npz/<class>/*.npz
- Uses pose_norm (T,25,3) -> time-series (T,D)
- Pads variable-length sequences per batch + uses pack_padded_sequence (recommended)
- Trains an LSTM classifier end-to-end on sequences (no shapelets)
- Saves hyperparameters + metrics + confusion matrix + classification report
  to: lstm_results/run_<timestamp>_seed<seed>.txt
- Saves checkpoint (.pt) with model weights + config for later use in report.ipynb

Only needs: numpy, torch, scikit-learn, tqdm
"""

from __future__ import annotations
from tqdm import tqdm

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Paths / labels
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_ROOT = PROJECT_ROOT / "data" / "poses_npz"
RESULTS_DIR = PROJECT_ROOT / "results" / "lstm_results" / "round_3"

LABELS: Dict[str, int] = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5,
}
INV_LABELS = {v: k for k, v in LABELS.items()}

SHORT_LABELS = {
    "boxing": "BX",
    "handclapping": "HC",
    "handwaving": "HW",
    "jogging": "JG",
    "running": "RN",
    "walking": "WK",
}


# SINGLE RUN HYPERPARAMS (start here)
RANDOM_SEED = 42
USE_CONF = True  # True => D=75, False => D=50

# Data split config (fixed split so experiments are comparable)
TEST_SIZE = 0.2
VAL_SPLIT = 0.15  # split from train only

# IMPORTANT: for final model training set this None
MAX_TRAIN_PER_CLASS: Optional[int] = 60  # None = use all train sequences

# LSTM model hyperparams (single run)
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
BIDIRECTIONAL = False
DROPOUT = 0.2  # applied between LSTM layers (if LSTM_LAYERS>1)

# Training hyperparams
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 20
GRAD_CLIP_NORM = 5.0

# Multi-run toggle (start False; once single run works, set True)
RUN_HYPERPARAM_MULTI_RUN = True

# Hyperparameter sweep (ONLY used if RUN_HYPERPARAM_MULTI_RUN=True)
# Keep list small; LSTM training is heavier than MLP.
HYPERPARAM_SWEEP = [
    # A) Current champion
    {"LSTM_HIDDEN": 64, "LSTM_LAYERS": 2, "BIDIRECTIONAL": True, "DROPOUT": 0.2, "LR": 1e-3, "BATCH_SIZE": 32},

    # B) Slightly less dropout (sometimes 0.2 is too strong)
    {"LSTM_HIDDEN": 64, "LSTM_LAYERS": 2, "BIDIRECTIONAL": True, "DROPOUT": 0.1, "LR": 1e-3, "BATCH_SIZE": 32},

    # C) Slightly more dropout (check regularization sweet spot)
    {"LSTM_HIDDEN": 64, "LSTM_LAYERS": 2, "BIDIRECTIONAL": True, "DROPOUT": 0.3, "LR": 1e-3, "BATCH_SIZE": 32},

    # D) Tiny LR decrease (for BiLSTM stability)
    {"LSTM_HIDDEN": 64, "LSTM_LAYERS": 2, "BIDIRECTIONAL": True, "DROPOUT": 0.2, "LR": 8e-4, "BATCH_SIZE": 32},

    # E) Tiny LR increase
    {"LSTM_HIDDEN": 64, "LSTM_LAYERS": 2, "BIDIRECTIONAL": True, "DROPOUT": 0.2, "LR": 1.2e-3, "BATCH_SIZE": 32},

    # F) Same model, larger batch (sometimes improves generalization)
    {"LSTM_HIDDEN": 64, "LSTM_LAYERS": 2, "BIDIRECTIONAL": True, "DROPOUT": 0.2, "LR": 1e-3, "BATCH_SIZE": 64},
]


# Seeding
def torch_set_seed(seed: int) -> None:
    """
    Strong seeding + deterministic settings.
    Note: exact determinism on GPU can still vary across driver/hardware,
    but this reduces run-to-run noise significantly.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce deterministic algorithms when possible
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    # for some CUDA ops determinism
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def seed_worker(worker_id: int) -> None:
    """
    Ensures dataloader workers are deterministically seeded.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Data loading
def list_npz_files() -> List[Path]:
    files: List[Path] = []
    for cls in LABELS.keys():
        cls_dir = NPZ_ROOT / cls
        if not cls_dir.exists():
            continue
        files.extend(sorted(cls_dir.glob("*.npz")))
    return files


def list_npz_toggle_check() -> List[Path]:
    all_files = list_npz_files()
    if not all_files:
        raise RuntimeError(f"No NPZ found under: {NPZ_ROOT}")
    bad = [p for p in all_files if p.parent.name not in LABELS]
    if bad:
        raise RuntimeError(f"Found NPZs in unknown class folders (check LABELS): e.g. {bad[0]}")
    return all_files


def load_npz_as_series(npz_path: Path, use_conf: bool = False) -> Tuple[np.ndarray, int]:
    """
    Returns:
      X: (T, D) float32
      y: int label
    """
    d = np.load(npz_path, allow_pickle=True)
    pose = d["pose_norm"].astype(np.float32)  # (T,25,3)
    y = int(d["label"])

    if use_conf:
        X = pose.reshape(pose.shape[0], 25 * 3)  # (T,75)
    else:
        X = pose[:, :, :2].reshape(pose.shape[0], 25 * 2)  # (T,50)

    return X, y


def load_dataset(file_paths: List[Path], use_conf: bool = False) -> Tuple[List[np.ndarray], np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for p in tqdm(file_paths, desc="Loading NPZ", unit="file"):
        X, y = load_npz_as_series(p, use_conf=use_conf)
        X_list.append(X)
        y_list.append(y)
    return X_list, np.array(y_list, dtype=np.int32)


def cap_train_per_class(
    X_list: List[np.ndarray],
    y: np.ndarray,
    cap: Optional[int],
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    if cap is None:
        return X_list, y

    rng = random.Random(seed)
    keep_idx: List[int] = []
    for c in sorted(set(y.tolist())):
        idx = np.where(y == c)[0].tolist()
        rng.shuffle(idx)
        keep_idx.extend(idx[:cap])
    keep_idx = sorted(keep_idx)

    X_new = [X_list[i] for i in keep_idx]
    y_new = y[keep_idx]
    return X_new, y_new


class PoseSeqDataset(Dataset):
    def __init__(self, X_list: List[np.ndarray], y: np.ndarray):
        self.X_list = X_list
        self.y = y

    def __len__(self) -> int:
        return len(self.X_list)

    def __getitem__(self, idx: int):
        x = self.X_list[idx]  # (T,D) np.float32
        y = int(self.y[idx])
        return x, y


def pad_collate(batch):
    """
    Pads to max length within batch.
    Returns:
      xb: (B, T_max, D)
      lengths: (B,)
      yb: (B,)
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    D = xs[0].shape[1]
    T_max = int(lengths.max().item())

    xb = torch.zeros((len(xs), T_max, D), dtype=torch.float32)
    for i, x in enumerate(xs):
        t = x.shape[0]
        xb[i, :t, :] = torch.from_numpy(x)

    yb = torch.tensor(ys, dtype=torch.long)
    return xb, lengths, yb


# Model
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )
        self.head = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Pack padded batch
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)

        # hn: (num_layers * num_directions, B, hidden)
        # Take last layer
        if self.bidirectional:
            # last layer has two directions at the end: [-2] (forward), [-1] (backward)
            h_last = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, 2*hidden)
        else:
            h_last = hn[-1]  # (B, hidden)

        logits = self.head(h_last)
        return logits


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (num_classes * counts)
    return w.astype(np.float32)


# Training / evaluation
def train_one_run(
    X_train_list: List[np.ndarray],
    y_train: np.ndarray,
    X_val_list: List[np.ndarray],
    y_val: np.ndarray,
    cfg: Dict,
    device: torch.device,
) -> Tuple[LSTMClassifier, Dict]:
    num_classes = len(INV_LABELS)
    input_dim = int(X_train_list[0].shape[1])

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_size=int(cfg["LSTM_HIDDEN"]),
        num_layers=int(cfg["LSTM_LAYERS"]),
        num_classes=num_classes,
        bidirectional=bool(cfg["BIDIRECTIONAL"]),
        dropout=float(cfg["DROPOUT"]),
    ).to(device)

    class_w = compute_class_weights(y_train, num_classes)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_w, device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["LR"]), weight_decay=WEIGHT_DECAY)

    # Dataloader determinism: generator + worker_init_fn
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_ds = PoseSeqDataset(X_train_list, y_train)
    val_ds = PoseSeqDataset(X_val_list, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["BATCH_SIZE"]),
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["BATCH_SIZE"]),
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    for ep in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for xb, lengths, yb in tqdm(train_loader, desc=f"Train ep {ep}/{MAX_EPOCHS}", leave=False):
            xb = xb.to(device=device, dtype=torch.float32)
            lengths = lengths.to(device=device)
            yb = yb.to(device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            logits = model(xb, lengths)
            loss = loss_fn(logits, yb)
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(xb.size(0))

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Val
        model.eval()
        v_loss = 0.0
        v_total = 0
        v_correct = 0

        with torch.no_grad():
            for xb, lengths, yb in tqdm(val_loader, desc=f"Val   ep {ep}/{MAX_EPOCHS}", leave=False):
                xb = xb.to(device=device, dtype=torch.float32)
                lengths = lengths.to(device=device)
                yb = yb.to(device=device, dtype=torch.long)

                logits = model(xb, lengths)
                loss = loss_fn(logits, yb)

                v_loss += float(loss.item()) * xb.size(0)
                pred = logits.argmax(dim=1)
                v_correct += int((pred == yb).sum().item())
                v_total += int(xb.size(0))

        val_loss = v_loss / max(1, v_total)
        val_acc = v_correct / max(1, v_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = (val_acc > best_val_acc + 1e-6)
        if improved:
            best_val_acc = val_acc
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "BEST_VAL_ACC": float(best_val_acc),
        "BEST_EPOCH": int(best_epoch),
        "EPOCHS_RUN": int(len(history["train_loss"])),
        "HISTORY": history,
    }
    return model, info


def predict(model: nn.Module, X_list: List[np.ndarray], batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    ds = PoseSeqDataset(X_list, np.zeros((len(X_list),), dtype=np.int32))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for xb, lengths, _ in tqdm(loader, desc="Predict", leave=False):
            xb = xb.to(device=device, dtype=torch.float32)
            lengths = lengths.to(device=device)
            logits = model(xb, lengths)
            pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int32)
            preds.append(pred)

    return np.concatenate(preds, axis=0)


# Results helpers
def per_class_table(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    lines = []
    header = f"{'class':<14} {'prec':>6} {'rec':>6} {'f1':>6} {'supp':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    rep = classification_report(
        y_true, y_pred,
        target_names=[INV_LABELS[i] for i in range(len(INV_LABELS))],
        output_dict=True,
        zero_division=0,
    )

    for i in range(len(INV_LABELS)):
        name = INV_LABELS[i]
        d = rep[name]
        lines.append(
            f"{name:<14} {d['precision']:>6.3f} {d['recall']:>6.3f} {d['f1-score']:>6.3f} {int(d['support']):>6}"
        )
    return "\n".join(lines)


def top_confusions(cm: np.ndarray, k: int = 8) -> str:
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cm[i, j] > 0:
                pairs.append((cm[i, j], i, j))
    pairs.sort(reverse=True)

    lines = []
    for cnt, i, j in pairs[:k]:
        lines.append(f"{INV_LABELS[i]} -> {INV_LABELS[j]} : {cnt}")
    return "\n".join(lines) if lines else "(no confusions)"


def pretty_confusion_matrix(cm: np.ndarray) -> str:
    labels = [INV_LABELS[i] for i in range(len(INV_LABELS))]
    short = [SHORT_LABELS[l] for l in labels]

    header = " " * 8 + "".join(f"{s:>6}" for s in short)
    lines = [header]

    for i, row in enumerate(cm):
        line = f"{short[i]:<6} |"
        for v in row:
            line += f"{v:>6}"
        lines.append(line)

    return "\n".join(lines)


def training_curves_table(history: Dict[str, List[float]]) -> str:
    tl = history["train_loss"]
    ta = history["train_acc"]
    vl = history["val_loss"]
    va = history["val_acc"]
    lines = []
    lines.append(f"{'epoch':>5} {'tr_loss':>10} {'tr_acc':>8} {'va_loss':>10} {'va_acc':>8}")
    lines.append("-" * 46)
    for i in range(len(tl)):
        ep = i + 1
        lines.append(f"{ep:>5} {tl[i]:>10.4f} {ta[i]:>8.4f} {vl[i]:>10.4f} {va[i]:>8.4f}")
    return "\n".join(lines)


def save_results_txt(
    run_path: Path,
    hyperparams: Dict,
    split_info: Dict,
    timings: Dict,
    acc: float,
    cm: np.ndarray,
    report: str,
) -> None:
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("w", encoding="utf-8") as f:
        f.write("=== LSTM (Pose Sequence Classification) ===\n \n")

        f.write("[Hyperparameters]\n")
        f.write(json.dumps(hyperparams, indent=2))
        f.write("\n \n")

        f.write("[Split / Data]\n")
        f.write(json.dumps(split_info, indent=2))
        f.write("\n \n")

        f.write("[Timings (seconds)]\n")
        f.write(json.dumps(timings, indent=2))
        f.write("\n \n")

        f.write(f"[Accuracy]\n{acc:.6f}\n\n")

        f.write("[Confusion Matrix]\n")
        f.write(pretty_confusion_matrix(cm))
        f.write("\n \n")

        f.write("[Classification Report]\n")
        f.write(report)
        f.write("\n")

        f.write(f"Macro F1: {hyperparams.get('MACRO_F1', 'NA')}\n")
        f.write(f"Weighted F1: {hyperparams.get('WEIGHTED_F1', 'NA')}\n \n")

        f.write("[Top Confusions]\n")
        f.write(hyperparams.get("TOP_CONFUSIONS", "(NA)"))
        f.write("\n \n")

        f.write("[Per-class Table]\n")
        f.write(hyperparams.get("PER_CLASS_TABLE", "(NA)"))
        f.write("\n \n")

        f.write("[Training Curves]\n")
        f.write(hyperparams.get("TRAINING_CURVES", "(NA)"))
        f.write("\n \n")



def labels_from_npz_paths(file_paths: List[Path]) -> np.ndarray:
    """Return y labels by reading each NPZ's internal label."""
    ys: List[int] = []
    for p in file_paths:
        d = np.load(p, allow_pickle=True)
        ys.append(int(d["label"]))
    return np.array(ys, dtype=np.int32)


def sanity_check_folder_vs_npz_labels(file_paths: List[Path]) -> None:
    """Optional: prints mismatches between folder label and NPZ label."""
    bad = []
    for p in file_paths:
        d = np.load(p, allow_pickle=True)
        y_npz = int(d["label"])
        y_folder = LABELS[p.parent.name]
        if y_npz != y_folder:
            bad.append((str(p), y_npz, y_folder))
    if bad:
        print(f"[WARNING] Found {len(bad)} label mismatches (NPZ vs folder). Example:")
        print("  path:", bad[0][0])
        print("  npz_label:", bad[0][1], "folder_label:", bad[0][2])
    else:
        print("[OK] No label mismatches between NPZ and folder.")


# Main
def main() -> None:
    torch_set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")

    all_files = list_npz_toggle_check()
    sanity_check_folder_vs_npz_labels(all_files)
    # Use NPZ labels as the source of truth for stratification (matches training labels)
    y_for_split = labels_from_npz_paths(all_files)

    # ONE fixed split for all runs
    train_files, test_files = train_test_split(
        all_files,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_for_split,
    )

    # Load full train/test sequences once per run (since USE_CONF can change between runs)
    runs = HYPERPARAM_SWEEP if RUN_HYPERPARAM_MULTI_RUN else [{
        "LSTM_HIDDEN": LSTM_HIDDEN,
        "LSTM_LAYERS": LSTM_LAYERS,
        "BIDIRECTIONAL": BIDIRECTIONAL,
        "DROPOUT": DROPOUT,
        "LR": LR,
        "BATCH_SIZE": BATCH_SIZE,
    }]

    print(f"Total files: {len(all_files)} | Train files: {len(train_files)} | Test files: {len(test_files)}")
    print(f"Planned runs: {len(runs)}")

    for run_idx, cfg in enumerate(runs, start=1):
        t0 = time.time()

        # Load sequences
        t_load0 = time.time()
        X_train_raw, y_train_raw = load_dataset(train_files, use_conf=USE_CONF)
        X_test_raw, y_test = load_dataset(test_files, use_conf=USE_CONF)
        t_load1 = time.time()

        # Cap training per class (speed)
        X_train_raw, y_train_raw = cap_train_per_class(
            X_train_raw, y_train_raw, cap=MAX_TRAIN_PER_CLASS, seed=RANDOM_SEED
        )

        # Train/val split (fixed)
        X_train_list, X_val_list, y_train, y_val = train_test_split(
            X_train_raw,
            y_train_raw,
            test_size=VAL_SPLIT,
            random_state=RANDOM_SEED,
            stratify=y_train_raw,
        )

        # Train LSTM
        t_fit0 = time.time()
        model, train_info = train_one_run(
            X_train_list=X_train_list,
            y_train=y_train,
            X_val_list=X_val_list,
            y_val=y_val,
            cfg=cfg,
            device=device,
        )
        t_fit1 = time.time()

        # Evaluate
        t_eval0 = time.time()
        pred = predict(model, X_test_raw, batch_size=int(cfg["BATCH_SIZE"]), device=device)
        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred)
        report = classification_report(
            y_test, pred,
            target_names=[INV_LABELS[i] for i in range(len(INV_LABELS))],
            digits=4,
            zero_division=0,
        )
        macro_f1 = f1_score(y_test, pred, average="macro")
        weighted_f1 = f1_score(y_test, pred, average="weighted")
        t_eval1 = time.time()
        t1 = time.time()

        print("\n============================================================")
        print(f"RUN {run_idx}/{len(runs)}")
        print(f"USE_CONF={USE_CONF} | cap={MAX_TRAIN_PER_CLASS if MAX_TRAIN_PER_CLASS is not None else 'ALL'}")
        print(f"LSTM hidden={cfg['LSTM_HIDDEN']} layers={cfg['LSTM_LAYERS']} bi={cfg['BIDIRECTIONAL']} drop={cfg['DROPOUT']}")
        print(f"LR={cfg['LR']} | BS={cfg['BATCH_SIZE']} | best_ep={train_info['BEST_EPOCH']}")
        print("------------------------------------------------------------")
        print(f"Accuracy     : {acc:.3f}")
        print(f"Macro F1     : {macro_f1:.3f}")
        print(f"Weighted F1  : {weighted_f1:.3f}")
        print("\n[Top confusions]")
        print(top_confusions(cm, k=10))

        # Save artifacts
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        tag = (
            f"run{run_idx:02d}"
            f"_conf{int(USE_CONF)}"
            f"_cap{MAX_TRAIN_PER_CLASS if MAX_TRAIN_PER_CLASS is not None else 'ALL'}"
            f"_h{cfg['LSTM_HIDDEN']}"
            f"_ly{cfg['LSTM_LAYERS']}"
            f"_bi{int(cfg['BIDIRECTIONAL'])}"
            f"_drop{cfg['DROPOUT']}"
            f"_lr{cfg['LR']:g}"
            f"_bs{cfg['BATCH_SIZE']}"
            f"_seed{RANDOM_SEED}"
        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        run_txt = RESULTS_DIR / f"{timestamp}_{tag}.txt"
        run_ckpt = RESULTS_DIR / f"{timestamp}_{tag}.pt"

        torch.save(
            {
                "state_dict": model.state_dict(),
                "cfg": cfg,
                "labels": LABELS,
                "inv_labels": INV_LABELS,
                "use_conf": USE_CONF,
                "random_seed": RANDOM_SEED,
                "max_train_per_class": MAX_TRAIN_PER_CLASS,
            },
            run_ckpt,
        )

        hyperparams = {
            "RANDOM_SEED": RANDOM_SEED,
            "USE_CONF": USE_CONF,
            "MAX_TRAIN_PER_CLASS": MAX_TRAIN_PER_CLASS,
            "test_size": TEST_SIZE,
            "val_split": VAL_SPLIT,
            "weight_decay": WEIGHT_DECAY,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "grad_clip_norm": GRAD_CLIP_NORM,

            "LSTM_HIDDEN": int(cfg["LSTM_HIDDEN"]),
            "LSTM_LAYERS": int(cfg["LSTM_LAYERS"]),
            "BIDIRECTIONAL": bool(cfg["BIDIRECTIONAL"]),
            "DROPOUT": float(cfg["DROPOUT"]),
            "LR": float(cfg["LR"]),
            "BATCH_SIZE": int(cfg["BATCH_SIZE"]),
            "MAX_EPOCHS": MAX_EPOCHS,

            "BEST_VAL_ACC": float(train_info["BEST_VAL_ACC"]),
            "BEST_EPOCH": int(train_info["BEST_EPOCH"]),
            "EPOCHS_RUN": int(train_info["EPOCHS_RUN"]),
            "CHECKPOINT_PATH": str(run_ckpt),

            "MACRO_F1": float(macro_f1),
            "WEIGHTED_F1": float(weighted_f1),
            "TOP_CONFUSIONS": top_confusions(cm, k=10),
            "PER_CLASS_TABLE": per_class_table(y_test, pred),
            "TRAINING_CURVES": training_curves_table(train_info["HISTORY"]),
        }

        split_info = {
            "total_files": len(all_files),
            "train_files_before_cap": len(train_files),
            "test_files": len(test_files),
            "train_sequences_after_cap": len(X_train_raw),
            "class_names": list(LABELS.keys()),
        }

        timings = {
            "load_data": round(t_load1 - t_load0, 4),
            "train_fit": round(t_fit1 - t_fit0, 4),
            "eval": round(t_eval1 - t_eval0, 4),
            "total": round(t1 - t0, 4),
        }

        save_results_txt(
            run_path=run_txt,
            hyperparams=hyperparams,
            split_info=split_info,
            timings=timings,
            acc=acc,
            cm=cm,
            report=report,
        )

        print(f"\nSaved results to: {run_txt}")
        print(f"Saved checkpoint to: {run_ckpt}")


if __name__ == "__main__":
    main()
