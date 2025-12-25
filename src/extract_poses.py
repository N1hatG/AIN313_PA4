"""
extract_poses.py

End-to-end extraction:
- Walks through data/raw_videos/<class>/**/*.avi
- Runs OpenPose (GPU) to write per-frame JSON
- Converts JSON -> pose arrays (T, 25, 3)
- Saves one .npz per video under data/poses_npz/<class>/

Fixes included:
✅ GPU usage ensured (explicit --num_gpu 1)
✅ Correct JSON frame ordering (sort by frame index)
✅ Unique output filenames (prevents overwriting)
✅ Resume-safe: validates existing NPZ; redoes if corrupted/partial
✅ Better errors + progress bar
✅ Optional: deletes per-video JSON after NPZ to save disk
✅ Optional: forward-fill empty detections for smoother sequences
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPOSE_ROOT = PROJECT_ROOT / "tools" / "openpose"
OPENPOSE_EXE = OPENPOSE_ROOT / "bin" / "OpenPoseDemo.exe"

DATASET_ROOT = PROJECT_ROOT / "data" / "raw_videos"
TMP_JSON_ROOT = PROJECT_ROOT / "data" / "_tmp_openpose_json"
OUT_NPZ_ROOT = PROJECT_ROOT / "data" / "poses_npz"

# -----------------------------
# Dataset labels (folder names must match)
# -----------------------------
LABELS: Dict[str, int] = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5,
}

VIDEO_EXTS = {".avi"}  # your dataset is .avi

# -----------------------------
# OpenPose settings
# -----------------------------
NET_RESOLUTION = "-1x256"       # speed/accuracy trade-off
NUMBER_PEOPLE_MAX = "1"         # pick one person

GPU_COUNT = "1"                 # force GPU usage
GPU_START = "0"

# -----------------------------
# Behavior toggles
# -----------------------------
DELETE_JSON_AFTER_NPZ = False     # saves a LOT of disk space
FORWARD_FILL_EMPTY = True        # smoother sequences if occasional missed detections

# -----------------------------
# Utilities
# -----------------------------
_FRAME_RE = re.compile(r"_(\d+)_keypoints\.json$")


def assert_paths() -> None:
    if not OPENPOSE_EXE.exists():
        raise FileNotFoundError(f"OpenPose exe not found: {OPENPOSE_EXE}")
    if not (OPENPOSE_ROOT / "models").exists():
        raise FileNotFoundError(f"OpenPose models folder not found: {OPENPOSE_ROOT / 'models'}")
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")


def list_videos() -> List[Tuple[str, Path]]:
    """Return list of (label_name, video_path)."""
    items: List[Tuple[str, Path]] = []
    for label_name in LABELS.keys():
        class_dir = DATASET_ROOT / label_name
        if not class_dir.exists():
            print(f"[WARN] Missing class folder: {class_dir} (skip)")
            continue
        for vp in class_dir.rglob("*"):
            if vp.suffix.lower() in VIDEO_EXTS:
                items.append((label_name, vp))
    return items


def make_unique_id(label_name: str, video_path: Path) -> str:
    """
    Build a unique, stable id for outputs based on relative path under class folder.
    Prevents overwriting if stems repeat.
    """
    rel = video_path.relative_to(DATASET_ROOT / label_name)  # can include subfolders
    safe = "_".join(rel.with_suffix("").parts)
    return safe


def run_openpose(video_path: Path, json_out_dir: Path) -> None:
    """Run OpenPose and write per-frame JSON keypoints into json_out_dir."""
    if json_out_dir.exists():
        shutil.rmtree(json_out_dir)
    json_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(OPENPOSE_EXE),
        "--model_pose", "BODY_25",
        "--video", str(video_path),
        "--display", "0",
        "--render_pose", "0",
        "--net_resolution", NET_RESOLUTION,
        "--write_json", str(json_out_dir),
        "--number_people_max", NUMBER_PEOPLE_MAX,
        "--process_real_time", "false",
        "--frame_step", "1",
        "--logging_level", "3",
        # Force GPU usage
        "--num_gpu", GPU_COUNT,
        "--num_gpu_start", GPU_START,
    ]

    # Must run from OpenPose root so relative "models/..." works.
    r = subprocess.run(cmd, cwd=str(OPENPOSE_ROOT), capture_output=True, text=True)

    if r.returncode != 0:
        raise RuntimeError(
            f"OpenPose failed for: {video_path}\n"
            f"STDERR:\n{r.stderr}\n"
            f"STDOUT:\n{r.stdout}\n"
        )


def read_body25_from_json(json_file: Path) -> np.ndarray:
    """
    Returns (25,3) [x,y,conf].
    If no people detected => zeros.
    If multiple people detected => choose the most confident (sum of conf).
    """
    data = json.loads(json_file.read_text(encoding="utf-8"))
    people = data.get("people", [])
    if not people:
        return np.zeros((25, 3), dtype=np.float32)

    best_pose = None
    best_conf = -1.0

    for p in people:
        arr = np.array(p.get("pose_keypoints_2d", []), dtype=np.float32)
        if arr.size != 25 * 3:
            continue
        pose = arr.reshape(25, 3)
        conf = float(pose[:, 2].sum())
        if conf > best_conf:
            best_conf = conf
            best_pose = pose

    if best_pose is None:
        return np.zeros((25, 3), dtype=np.float32)

    return best_pose.astype(np.float32)


def normalize_pose(seq: np.ndarray) -> np.ndarray:
    """
    Simple normalization:
      - Center by MidHip (index 8)
      - Scale by shoulder distance (RShoulder=2, LShoulder=5)
    Input: seq (T,25,3)
    Output: seq_norm (T,25,3) with x,y normalized; conf unchanged.
    """
    seq = seq.copy()
    midhip, rsh, lsh = 8, 2, 5

    origin = seq[:, midhip, :2]  # (T,2)
    seq[:, :, 0] -= origin[:, 0:1]
    seq[:, :, 1] -= origin[:, 1:2]

    shoulder_vec = seq[:, lsh, :2] - seq[:, rsh, :2]
    dist = np.linalg.norm(shoulder_vec, axis=1)
    dist = np.where(dist < 1e-6, 1.0, dist)

    seq[:, :, 0] /= dist[:, None]
    seq[:, :, 1] /= dist[:, None]
    return seq


def frame_id(p: Path) -> int:
    """
    Extract frame index from filename like:
    ..._000000000123_keypoints.json
    """
    m = _FRAME_RE.search(p.name)
    return int(m.group(1)) if m else 10**18


def forward_fill_empty_frames(pose: np.ndarray) -> np.ndarray:
    """
    If a frame has zero confidence everywhere (no detection),
    copy the previous frame (very common minor fix for stability).
    """
    if pose.shape[0] <= 1:
        return pose
    pose = pose.copy()
    for t in range(1, pose.shape[0]):
        if float(pose[t, :, 2].sum()) == 0.0 and float(pose[t - 1, :, 2].sum()) > 0.0:
            pose[t] = pose[t - 1]
    return pose


def json_dir_to_npz(json_dir: Path, out_npz: Path, label_name: str, video_path: Path) -> None:
    json_files = sorted(json_dir.glob("*.json"), key=frame_id)
    if not json_files:
        raise RuntimeError(f"No JSON files produced in: {json_dir}")

    frames = [read_body25_from_json(jf) for jf in json_files]
    pose = np.stack(frames, axis=0)  # (T,25,3)

    if FORWARD_FILL_EMPTY:
        pose = forward_fill_empty_frames(pose)

    pose_norm = normalize_pose(pose)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        pose=pose.astype(np.float32),
        pose_norm=pose_norm.astype(np.float32),
        frame_idx=np.arange(pose.shape[0], dtype=np.int32),
        label=np.int32(LABELS[label_name]),
        label_name=label_name,
        video_path=str(video_path),
    )


def npz_is_valid(npz_path: Path) -> bool:
    """Check if an existing NPZ looks valid (resume-safe)."""
    try:
        d = np.load(npz_path, allow_pickle=True)
        pose = d["pose"]
        if pose.ndim != 3 or pose.shape[1:] != (25, 3) or pose.shape[0] < 1:
            return False
        # basic key presence
        _ = d["pose_norm"]
        _ = d["label"]
        return True
    except Exception:
        return False


def main() -> None:
    assert_paths()

    TMP_JSON_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_NPZ_ROOT.mkdir(parents=True, exist_ok=True)

    videos = list_videos()
    if not videos:
        raise RuntimeError(f"No videos found under: {DATASET_ROOT}")

    print(f"Found {len(videos)} videos total.")
    print("Extracting BODY_25 → JSON → NPZ (GPU)...")

    for label_name, vp in tqdm(videos, desc="Videos", unit="video"):
        uid = make_unique_id(label_name, vp)

        out_npz = OUT_NPZ_ROOT / label_name / f"{uid}.npz"
        tmp_json = TMP_JSON_ROOT / label_name / uid

        # Resume-safe skip
        if out_npz.exists() and npz_is_valid(out_npz):
            continue
        elif out_npz.exists() and not npz_is_valid(out_npz):
            out_npz.unlink(missing_ok=True)

        run_openpose(vp, tmp_json)
        json_dir_to_npz(tmp_json, out_npz, label_name, vp)

        # Save disk space (optional but recommended)
        if DELETE_JSON_AFTER_NPZ:
            shutil.rmtree(tmp_json, ignore_errors=True)

    print("\nDONE.")
    print("NPZ features saved under:", OUT_NPZ_ROOT)
    if DELETE_JSON_AFTER_NPZ:
        print("Temporary JSON was deleted per video to save space.")
    else:
        print("Temporary JSON kept under:", TMP_JSON_ROOT)
        print("After verifying NPZs, you can delete:", TMP_JSON_ROOT)


if __name__ == "__main__":
    main()
