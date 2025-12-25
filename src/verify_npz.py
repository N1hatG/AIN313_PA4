from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NPZ_ROOT = ROOT / "data" / "poses_npz"

def main():
    files = list(NPZ_ROOT.rglob("*.npz"))
    print("NPZ count:", len(files))
    if not files:
        return

    f = files[0]
    d = np.load(f, allow_pickle=True)
    print("Sample file:", f)
    print("keys:", d.files)
    print("pose shape:", d["pose"].shape)          # (T,25,3)
    print("pose_norm shape:", d["pose_norm"].shape)
    print("label:", int(d["label"]), "label_name:", str(d["label_name"]))
    print("video_path:", str(d["video_path"]))
    
    f = next(Path("data/poses_npz").rglob("*.npz"))
    d = np.load(f, allow_pickle=True)
    pose = d["pose"]

    zero_frames = (pose[:,:,2].sum(axis=1) == 0).mean()
    print("Empty-frame ratio:", zero_frames)

    print("Conf min/mean/max:", pose[:,:,2].min(), pose[:,:,2].mean(), pose[:,:,2].max())


if __name__ == "__main__":
    main()
