"""
shapelets_mlp.py

Random Shapelet Transform + MLP classifier (QUICK TEST RUN CONFIG)
- Loads NPZ files from data/poses_npz/<class>/*.npz
- Uses pose_norm (T,25,3) -> time-series (T,D)
- Samples random shapelets from TRAIN ONLY
- Computes min z-normalized Euclidean distance over each shapelet => features
- Trains sklearn MLPClassifier
- Saves hyperparameters + metrics + confusion matrix + classification report
  to: shapelets_mlp_results/run_<timestamp>_seed<seed>.txt

Only needs: numpy, scikit-learn, tqdm
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset



# Paths / labels
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_ROOT = PROJECT_ROOT / "data" / "poses_npz"
RESULTS_DIR = PROJECT_ROOT / "shapelets_mlp_results" / "based_on_20251226_214457_run05_K64_L20-80_cap60_conf1_mlp256x128_lr0.001_ep120_seed42.pt"

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


# SIGNLE RUN HYPERPARAMS
RANDOM_SEED = 42

NUM_SHAPELETS = 128
LEN_MIN = 25
LEN_MAX = 60
MAX_TRAIN_PER_CLASS = 100  # cap train size for speed; set None for full run later

MLP_HIDDEN = (256, 128)
MLP_MAX_ITER = 80
MLP_LR = 1e-3

USE_CONF = False  # keep False for speed (D=50). True => D=75
RUN_HYPERPARAM_MULTI_RUN = True #True runs multi-run test, False runs with SINGLE RUN PARAMS ABOVE

# Hyperparameter multi-run configs
# Keep this list small to avoid long runtime. or the runnnsss foreveeeerrrr
HYPERPARAM_SWEEP = [
    {"NUM_SHAPELETS": 32,  "LEN_MIN": 20, "LEN_MAX": 80, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},

    {"NUM_SHAPELETS": 48,  "LEN_MIN": 20, "LEN_MAX": 80, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},

    {"NUM_SHAPELETS": 64,  "LEN_MIN": 20, "LEN_MAX": 80, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},

    {"NUM_SHAPELETS": 96,  "LEN_MIN": 20, "LEN_MAX": 80, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},

    {"NUM_SHAPELETS": 128, "LEN_MIN": 20, "LEN_MAX": 80, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},

    {"NUM_SHAPELETS": 64,  "LEN_MIN": 15, "LEN_MAX": 45, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},

    {"NUM_SHAPELETS": 64,  "LEN_MIN": 20, "LEN_MAX": 50, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},

    {"NUM_SHAPELETS": 64,  "LEN_MIN": 25, "LEN_MAX": 55, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},

    {"NUM_SHAPELETS": 64,  "LEN_MIN": 30, "LEN_MAX": 70, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},

    {"NUM_SHAPELETS": 64,  "LEN_MIN": 40, "LEN_MAX": 80, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},

    {"NUM_SHAPELETS": 96,  "LEN_MIN": 20, "LEN_MAX": 50, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},

    {"NUM_SHAPELETS": 96,  "LEN_MIN": 25, "LEN_MAX": 55, "MAX_TRAIN_PER_CLASS": 60,
     "MLP_HIDDEN": (256, 128), "MLP_LR": 1e-3, "MLP_MAX_ITER": 120, "USE_CONF": True},
]

_SINGLE_RUN_CFG = {
    "NUM_SHAPELETS": NUM_SHAPELETS,
    "LEN_MIN": LEN_MIN,
    "LEN_MAX": LEN_MAX,
    "MAX_TRAIN_PER_CLASS": MAX_TRAIN_PER_CLASS,
    "MLP_HIDDEN": MLP_HIDDEN,
    "MLP_LR": MLP_LR,
    "MLP_MAX_ITER": MLP_MAX_ITER,
    "USE_CONF": USE_CONF,
}


BATCH_SIZE = 128
VAL_SPLIT = 0.15
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
EARLY_STOPPING_PATIENCE = 12
GRAD_CLIP_NORM = 5.0


# Utilities
def list_npz_files() -> List[Path]:
    files: List[Path] = []
    for cls in LABELS.keys():
        cls_dir = NPZ_ROOT / cls
        if not cls_dir.exists():
            continue
        files.extend(sorted(cls_dir.glob("*.npz")))
    return files


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


# Random Shapelets
@dataclass(frozen=True)
class Shapelet:
    pattern: np.ndarray  # (L, D) z-normalized
    length: int


def z_norm(ts: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-normalize each dimension independently over time: ts (L, D)."""
    mu = ts.mean(axis=0, keepdims=True)
    sd = ts.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (ts - mu) / sd


def min_dist_to_shapelet(series: np.ndarray, shapelet: Shapelet, eps: float = 1e-8) -> float:
    """
    Vectorized version of min z-normalized Euclidean distance.
    Same definition as before:
      - For each window: z-norm per dimension over time
      - dist = ||window - shapelet|| / L
      - return min dist over all windows
    """
    T, D = series.shape
    L = shapelet.length

    if T < L:
        pad = np.repeat(series[-1:, :], repeats=(L - T), axis=0)
        series = np.concatenate([series, pad], axis=0)
        T = L

    windows = np.lib.stride_tricks.sliding_window_view(series, window_shape=(L, D))
    windows = windows.reshape(-1, L, D)

    mu = windows.mean(axis=1, keepdims=True)
    sd = windows.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    windows_zn = (windows - mu) / sd

    diff = windows_zn - shapelet.pattern[None, :, :]
    dist = np.linalg.norm(diff, axis=(1, 2)) / L
    return float(dist.min())

def sample_shapelets(
    X_train: List[np.ndarray],
    y_train: np.ndarray,
    num_shapelets: int,
    len_min: int,
    len_max: int,
    seed: int = 42,
) -> List[Shapelet]:
    """Balanced-ish sampling across classes."""
    rng = random.Random(seed)

    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(y_train.tolist()):
        by_class.setdefault(y, []).append(i)

    classes = sorted(by_class.keys())
    shapelets: List[Shapelet] = []

    for k in range(num_shapelets):
        cls = classes[k % len(classes)]
        idx = rng.choice(by_class[cls])
        series = X_train[idx]
        T = series.shape[0]

        if T <= 1:
            continue

        if T < len_min:
            L = T
        else:
            L = rng.randint(len_min, min(len_max, T))

        if L <= 1:
            continue

        start = rng.randint(0, max(0, T - L))
        pat = series[start:start + L, :]
        pat = z_norm(pat).astype(np.float32)

        shapelets.append(Shapelet(pattern=pat, length=L))

    return shapelets


def transform_with_shapelets(
    X: List[np.ndarray],
    shapelets: List[Shapelet],
    show_progress: bool = True,
) -> np.ndarray:
    """List of variable-length series -> (N, K) feature matrix."""
    N = len(X)
    K = len(shapelets)
    feats = np.zeros((N, K), dtype=np.float32)

    it = range(N)
    if show_progress:
        it = tqdm(it, desc="Shapelet transform", unit="seq")

    for i in it:
        series = X[i]
        for j, sh in enumerate(shapelets):
            feats[i, j] = min_dist_to_shapelet(series, sh)

    return feats


class TorchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def torch_set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def standardize_fit(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32)


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (num_classes * counts)
    return w.astype(np.float32)


def train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden: Tuple[int, ...],
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> Tuple[TorchMLP, Dict]:
    in_dim = int(X_train.shape[1])
    out_dim = int(len(INV_LABELS))

    model = TorchMLP(in_dim, hidden=hidden, out_dim=out_dim, dropout=DROPOUT).to(device)

    class_w = compute_class_weights(y_train, out_dim)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_w, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
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

        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.long)
                logits = model(xb)
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


def torch_predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            logits = model(xb)
            pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int32)
            out.append(pred)
    return np.concatenate(out, axis=0)


# Results logging
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
        f.write("=== Shapelets + MLP (Random Shapelet Transform) ===\n \n")

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



def main() -> None:
    torch_set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")

    all_files = list_npz_toggle_check()
    y_for_split = np.array([LABELS[p.parent.name] for p in all_files], dtype=np.int32)

    # IMPORTANT: use ONE fixed split for all runs
    train_files, test_files = train_test_split(
        all_files,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_for_split,
    )

    runs = HYPERPARAM_SWEEP if RUN_HYPERPARAM_MULTI_RUN else [_SINGLE_RUN_CFG]

    print(f"Total files: {len(all_files)} | Train files: {len(train_files)} | Test files: {len(test_files)}")
    print(f"Planned sweep runs: {len(runs)}")

    for run_idx, cfg in enumerate(runs, start=1):
        NUM_SHAPELETS = int(cfg["NUM_SHAPELETS"])
        LEN_MIN = int(cfg["LEN_MIN"])
        LEN_MAX = int(cfg["LEN_MAX"])
        MAX_TRAIN_PER_CLASS = cfg["MAX_TRAIN_PER_CLASS"]
        MLP_HIDDEN = tuple(cfg["MLP_HIDDEN"])
        MLP_LR = float(cfg["MLP_LR"])
        MLP_MAX_ITER = int(cfg["MLP_MAX_ITER"])
        USE_CONF = bool(cfg["USE_CONF"])

        t0 = time.time()

        # Load sequences
        t_load0 = time.time()
        X_train_raw, y_train_raw = load_dataset(train_files, use_conf=USE_CONF)
        X_test_raw, y_test = load_dataset(test_files, use_conf=USE_CONF)
        t_load1 = time.time()

        # Cap training per class (speed)
        if MAX_TRAIN_PER_CLASS is not None:
            keep_idx: List[int] = []
            for c in sorted(set(y_train_raw.tolist())):
                idx = np.where(y_train_raw == c)[0].tolist()
                random.shuffle(idx)
                keep_idx.extend(idx[:MAX_TRAIN_PER_CLASS])
            keep_idx = sorted(keep_idx)
            X_train_raw = [X_train_raw[i] for i in keep_idx]
            y_train_raw = y_train_raw[keep_idx]

        # Sample shapelets on TRAIN ONLY
        t_samp0 = time.time()
        shapelets = sample_shapelets(
            X_train_raw, y_train_raw,
            num_shapelets=NUM_SHAPELETS,
            len_min=LEN_MIN,
            len_max=LEN_MAX,
            seed=RANDOM_SEED,
        )
        t_samp1 = time.time()

        # Transform
        t_tr0 = time.time()
        Xtr_feat_all = transform_with_shapelets(X_train_raw, shapelets, show_progress=True)
        t_tr1 = time.time()

        t_te0 = time.time()
        Xte_feat = transform_with_shapelets(X_test_raw, shapelets, show_progress=True)
        t_te1 = time.time()

        Xtr_feat, Xval_feat, y_train, y_val = train_test_split(
            Xtr_feat_all,
            y_train_raw,
            test_size=VAL_SPLIT,
            random_state=RANDOM_SEED,
            stratify=y_train_raw,
        )

        mu, sd = standardize_fit(Xtr_feat)
        Xtr_feat = standardize_apply(Xtr_feat, mu, sd)
        Xval_feat = standardize_apply(Xval_feat, mu, sd)
        Xte_feat = standardize_apply(Xte_feat, mu, sd)

        # Train MLP
        t_fit0 = time.time()
        model, train_info = train_torch_mlp(
            X_train=Xtr_feat,
            y_train=y_train,
            X_val=Xval_feat,
            y_val=y_val,
            hidden=MLP_HIDDEN,
            lr=MLP_LR,
            epochs=MLP_MAX_ITER,
            seed=RANDOM_SEED,
            device=device,
        )
        t_fit1 = time.time()

        # Evaluate
        t_eval0 = time.time()
        pred = torch_predict(model, Xte_feat, device=device)
        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred)
        report = classification_report(
            y_test, pred,
            target_names=[INV_LABELS[i] for i in range(len(INV_LABELS))],
            digits=4,
            zero_division=0,
        )
        t_eval1 = time.time()
        t1 = time.time()

        macro_f1 = f1_score(y_test, pred, average="macro")
        weighted_f1 = f1_score(y_test, pred, average="weighted")

        print("\n============================================================")
        print(f"RUN {run_idx}/{len(runs)}")
        print(f"K={NUM_SHAPELETS} | L={LEN_MIN}-{LEN_MAX} | cap={MAX_TRAIN_PER_CLASS} | conf={USE_CONF}")
        print(f"MLP={MLP_HIDDEN} | lr={MLP_LR} | ep={MLP_MAX_ITER} | best_ep={train_info['BEST_EPOCH']}")
        print("------------------------------------------------------------")
        print(f"Accuracy     : {acc:.3f}")
        print(f"Macro F1     : {macro_f1:.3f}")
        print(f"Weighted F1  : {weighted_f1:.3f}")
        print("\n[Top confusions]")
        print(top_confusions(cm, k=10))

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        tag = (
            f"run{run_idx:02d}"
            f"_K{NUM_SHAPELETS}"
            f"_L{LEN_MIN}-{LEN_MAX}"
            f"_cap{MAX_TRAIN_PER_CLASS if MAX_TRAIN_PER_CLASS is not None else 'ALL'}"
            f"_conf{int(USE_CONF)}"
            f"_mlp{'x'.join(map(str, MLP_HIDDEN))}"
            f"_lr{MLP_LR:g}"
            f"_ep{MLP_MAX_ITER}"
            f"_seed{RANDOM_SEED}"
        )
        run_txt = RESULTS_DIR / f"{timestamp}_{tag}.txt"
        run_ckpt = RESULTS_DIR / f"{timestamp}_{tag}.pt"

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mu": mu,
                "sd": sd,
                "cfg": cfg,
                "labels": LABELS,
                "inv_labels": INV_LABELS,
            },
            run_ckpt,
        )

        hyperparams = {
            "RANDOM_SEED": RANDOM_SEED,
            "USE_CONF": USE_CONF,
            "NUM_SHAPELETS": NUM_SHAPELETS,
            "LEN_MIN": LEN_MIN,
            "LEN_MAX": LEN_MAX,
            "MAX_TRAIN_PER_CLASS": MAX_TRAIN_PER_CLASS,
            "MLP_HIDDEN": list(MLP_HIDDEN),
            "MLP_MAX_ITER": MLP_MAX_ITER,
            "MLP_LR": MLP_LR,
            "test_size": 0.2,
            "val_split": VAL_SPLIT,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
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
            "sample_shapelets": round(t_samp1 - t_samp0, 4),
            "transform_train": round(t_tr1 - t_tr0, 4),
            "transform_test": round(t_te1 - t_te0, 4),
            "mlp_fit": round(t_fit1 - t_fit0, 4),
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


def list_npz_toggle_check() -> List[Path]:
    all_files = list_npz_files()
    if not all_files:
        raise RuntimeError(f"No NPZ found under: {NPZ_ROOT}")
    # quick sanity: ensure folder names match LABELS
    bad = [p for p in all_files if p.parent.name not in LABELS]
    if bad:
        raise RuntimeError(f"Found NPZs in unknown class folders (check LABELS): e.g. {bad[0]}")
    return all_files


#Helper functions for printing and saving reuslts
def format_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    r = s - 60 * m
    return f"{m}m {r:.1f}s"


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
    """
    Return top-k off-diagonal confusions like:
    jogging -> running : 11
    """
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


if __name__ == "__main__":
    main()
