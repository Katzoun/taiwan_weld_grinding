"""
PointCNN inference: take a preprocessed .ply (output of Point_Cloud_Preprocessing.ipynb)
and produce a predicted_3.txt-style file (x, y, z, label) for
Weld_Recognition&Grinding_Path_Planning(PointCNN).ipynb.

Mirrors exactly how WireHarenessDataset builds a training sample
(PointCNN_Training_Functions/dataset/wireharness_dataset.py): farthest-point
sample down to num_points if the cloud is larger, or pad by repeating the
first points if smaller. The model was trained on single FPS-reduced samples,
not sliding-window crops of the full scan.

pointcnn_seg_1024.py (imported by PointCNN_Training.ipynb) is not present
anywhere in this repo. PointCNN_Training_Functions/model/pointcnn_seg.py's
PointCNNSeg class is architecturally identical -- verified by loading
pointcnn-best_8_2048.ckpt with it (state_dict keys match exactly, no
missing/unexpected keys) -- and is used here instead.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch_geometric.nn import fps

sys.path.insert(0, str(Path(__file__).parent / "PointCNN_Training_Functions"))
from model.pointcnn_seg import PointCNNSeg  # noqa: E402


def load_model(ckpt_path, num_classes=2, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PointCNNSeg.load_from_checkpoint(
        ckpt_path, map_location=device, num_classes=num_classes, weight_balance=None
    )
    model.eval()
    model.to(device)
    return model, device


def sample_to_num_points(coords, num_points):
    if coords.shape[0] > num_points:
        idx = fps(coords, ratio=num_points / coords.shape[0])
        idx = idx[:num_points]
        return coords[idx]
    pad = num_points - coords.shape[0]
    return torch.cat([coords, coords[:pad]], dim=0)


def run_inference(ply_path, ckpt_path, out_path, num_points=2048, num_classes=2, passes=1,
                   vote_threshold=0.5, batch_size=16):
    """
    The model was trained on single 2048-point FPS samples (see module
    docstring), so one pass only "sees" ~2% of a 100k-point real scan --
    fine for a quick check, too sparse to build a path from. fps() picks a
    random starting point by default (torch_geometric.nn.fps random_start=True),
    so repeated passes sample different subsets of the same real point cloud.
    Voting the per-point predictions across passes (not fabricating any new
    points -- only ever labeling points that are actually in the scan) gives
    much denser coverage of the true bead than a single pass. Passes are run
    batch_size at a time so the GPU isn't fed one sample at a time.
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    coords = np.asarray(pcd.points, dtype=np.float32)
    if len(coords) == 0:
        raise ValueError(f"{ply_path} has no points")
    coords_t = torch.tensor(coords)
    n_total = coords_t.shape[0]

    model, device = load_model(ckpt_path, num_classes=num_classes)

    bead_votes = np.zeros(n_total, dtype=np.int32)
    seen_votes = np.zeros(n_total, dtype=np.int32)

    t0 = time.time()
    done = 0
    while done < passes:
        b = min(batch_size, passes - done)
        idx_list = []
        sample_list = []
        for _ in range(b):
            if n_total > num_points:
                idx = fps(coords_t, ratio=num_points / n_total)[:num_points]
                sample_list.append(coords_t[idx])
            else:
                idx = torch.arange(n_total)
                sample_list.append(sample_to_num_points(coords_t, num_points))
            idx_list.append(idx)

        batch = torch.stack(sample_list, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)  # (b, num_classes, num_points)
            pred = torch.argmax(logits, dim=1).cpu().numpy()  # (b, num_points)

        for i, idx in enumerate(idx_list):
            idx_np = idx.numpy()[: pred.shape[1]]
            seen_votes[idx_np] += 1
            bead_votes[idx_np] += (pred[i] == 1).astype(np.int32)

        done += b
    t1 = time.time()

    seen_mask = seen_votes > 0
    label = np.zeros(n_total, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        frac = np.where(seen_mask, bead_votes / np.maximum(seen_votes, 1), 0.0)
    label[seen_mask & (frac >= vote_threshold)] = 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.column_stack([coords[seen_mask], label[seen_mask]])
    np.savetxt(out_path, out, fmt="%.6f")

    n_bead = int((label[seen_mask] == 1).sum())
    n_seen = int(seen_mask.sum())
    print(f"Device: {device}")
    print(f"{passes} pass(es), {t1 - t0:.3f}s total")
    print(f"Points covered: {n_seen} / {n_total} ({100 * n_seen / n_total:.1f}%)")
    print(f"Weld bead points: {n_bead} / {n_seen}")
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply_path", help="Preprocessed point cloud (.ply)")
    parser.add_argument(
        "--ckpt",
        default="PointCNN_weight(Torch)/pointcnn-best_8_2048.ckpt",
        help="Checkpoint path (default: best val-accuracy checkpoint, 2048 pts/batch 8)",
    )
    parser.add_argument("--out", default="outputs/predicted_3.txt")
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument(
        "--passes",
        type=int,
        default=60,
        help="Number of FPS re-samples to vote over (denser coverage on scans >> num_points; default 60)",
    )
    parser.add_argument("--vote-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16, help="Passes per GPU forward call")
    args = parser.parse_args()

    run_inference(
        args.ply_path, args.ckpt, args.out,
        num_points=args.num_points, passes=args.passes, vote_threshold=args.vote_threshold,
        batch_size=args.batch_size,
    )
