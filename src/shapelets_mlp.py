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
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score



# Paths / labels
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_ROOT = PROJECT_ROOT / "data" / "poses_npz"
RESULTS_DIR = PROJECT_ROOT / "shapelets_mlp_results"

LABELS: Dict[str, int] = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5,
}
INV_LABELS = {v: k for k, v in LABELS.items()}


# HYPERPARAMS
RANDOM_SEED = 42

NUM_SHAPELETS = 64
LEN_MIN = 20
LEN_MAX = 80
MAX_TRAIN_PER_CLASS = 60  # cap train size for speed; set None for full run later

MLP_HIDDEN = (256, 128)
MLP_MAX_ITER = 80
MLP_LR = 1e-3

USE_CONF = False  # keep False for speed (D=50). True => D=75


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

    # windows: (T-L+1, L, D)
    windows = np.lib.stride_tricks.sliding_window_view(series, window_shape=(L, D))
    # sliding_window_view gives (T-L+1, 1, L, D) for 2D input sometimes depending on numpy, so:
    windows = windows.reshape(-1, L, D)

    # z-norm each window per dimension
    mu = windows.mean(axis=1, keepdims=True)            # (W,1,D)
    sd = windows.std(axis=1, keepdims=True)             # (W,1,D)
    sd = np.where(sd < eps, 1.0, sd)
    windows_zn = (windows - mu) / sd

    # shapelet.pattern: (L,D)
    diff = windows_zn - shapelet.pattern[None, :, :]    # (W,L,D)
    # Euclidean norm over (L,D)
    dist = np.linalg.norm(diff, axis=(1, 2)) / L        # (W,)
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

        # choose L within [len_min, len_max] but not exceeding T
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
        f.write("=== Shapelets + MLP (Random Shapelet Transform) ===\n\n")

        f.write("[Hyperparameters]\n")
        f.write(json.dumps(hyperparams, indent=2))
        f.write("\n\n")

        f.write("[Split / Data]\n")
        f.write(json.dumps(split_info, indent=2))
        f.write("\n\n")

        f.write("[Timings (seconds)]\n")
        f.write(json.dumps(timings, indent=2))
        f.write("\n\n")

        f.write(f"[Accuracy]\n{acc:.6f}\n\n")

        f.write("[Confusion Matrix]\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("[Classification Report]\n")
        f.write(report)
        f.write("\n")

        f.write(f"Macro F1: {hyperparams.get('MACRO_F1', 'NA')}\n")
        f.write(f"Weighted F1: {hyperparams.get('WEIGHTED_F1', 'NA')}\n\n")

        f.write("[Top Confusions]\n")
        f.write(hyperparams.get("TOP_CONFUSIONS", "(NA)"))
        f.write("\n\n")

        f.write("[Per-class Table]\n")
        f.write(hyperparams.get("PER_CLASS_TABLE", "(NA)"))
        f.write("\n\n")



def main() -> None:
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    all_files = list_npz_toggle_check()
    y_for_split = np.array([LABELS[p.parent.name] for p in all_files], dtype=np.int32)

    t0 = time.time()
    train_files, test_files = train_test_split(
        all_files,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_for_split,
    )

    # Load sequences
    t_load0 = time.time()
    X_train, y_train = load_dataset(train_files, use_conf=USE_CONF)
    X_test, y_test = load_dataset(test_files, use_conf=USE_CONF)
    t_load1 = time.time()

    # Cap training per class (speed)
    if MAX_TRAIN_PER_CLASS is not None:
        keep_idx: List[int] = []
        for c in sorted(set(y_train.tolist())):
            idx = np.where(y_train == c)[0].tolist()
            random.shuffle(idx)
            keep_idx.extend(idx[:MAX_TRAIN_PER_CLASS])
        keep_idx = sorted(keep_idx)
        X_train = [X_train[i] for i in keep_idx]
        y_train = y_train[keep_idx]

    # Sample shapelets on TRAIN ONLY
    t_samp0 = time.time()
    shapelets = sample_shapelets(
        X_train, y_train,
        num_shapelets=NUM_SHAPELETS,
        len_min=LEN_MIN,
        len_max=LEN_MAX,
        seed=RANDOM_SEED,
    )
    t_samp1 = time.time()

    # Transform
    t_tr0 = time.time()
    Xtr_feat = transform_with_shapelets(X_train, shapelets, show_progress=True)
    t_tr1 = time.time()

    t_te0 = time.time()
    Xte_feat = transform_with_shapelets(X_test, shapelets, show_progress=True)
    t_te1 = time.time()

    # Train MLP
    t_fit0 = time.time()
    clf = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN,
        activation="relu",
        solver="adam",
        learning_rate_init=MLP_LR,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    clf.fit(Xtr_feat, y_train)
    t_fit1 = time.time()

    # Evaluate
    t_eval0 = time.time()
    pred = clf.predict(Xte_feat)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(
        y_test, pred,
        target_names=[INV_LABELS[i] for i in range(len(INV_LABELS))],
        digits=4,
    )
    t_eval1 = time.time()
    t1 = time.time()

    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")

    print("\n=== Shapelets + MLP (Readable Summary) ===")
    print(f"Accuracy     : {acc:.3f}")
    print(f"Macro F1     : {macro_f1:.3f}")
    print(f"Weighted F1  : {weighted_f1:.3f}")

    print("\n[Per-class]")
    print(per_class_table(y_test, pred))

    print("\n[Top confusions]")
    print(top_confusions(cm, k=10))

    print("\n[Timings]")
    print(f"Load data       : {format_seconds(t_load1 - t_load0)}")
    print(f"Sample shapelets: {format_seconds(t_samp1 - t_samp0)}")
    print(f"Transform train : {format_seconds(t_tr1 - t_tr0)}")
    print(f"Transform test  : {format_seconds(t_te1 - t_te0)}")
    print(f"MLP fit         : {format_seconds(t_fit1 - t_fit0)}")
    print(f"Eval            : {format_seconds(t_eval1 - t_eval0)}")
    print(f"TOTAL           : {format_seconds(t1 - t0)}")


    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_txt = RESULTS_DIR / f"run_{timestamp}_seed{RANDOM_SEED}.txt"

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

        # Extra readable metrics/text blobs
        "MACRO_F1": float(macro_f1),
        "WEIGHTED_F1": float(weighted_f1),
        "TOP_CONFUSIONS": top_confusions(cm, k=10),
        "PER_CLASS_TABLE": per_class_table(y_test, pred),
    }
    split_info = {
        "total_files": len(all_files),
        "train_files_before_cap": len(train_files),
        "test_files": len(test_files),
        "train_sequences_after_cap": len(X_train),
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

    # compute per-class metrics manually via report dict
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


if __name__ == "__main__":
    main()
