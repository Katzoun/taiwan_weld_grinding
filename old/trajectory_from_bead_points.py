"""
Build a grinding-path trajectory (position + normal + quaternion per point)
directly from sparse PointCNN bead predictions, without the image
rasterize/skeletonize step used in Weld_Recognition&Grinding_Path_Planning
(PointCNN).ipynb.

Why: that notebook rasterizes the bead points into a 1px=1mm image and
erodes/dilates/skeletonizes it -- it needs a solid, densely-populated blob.
pointcnn_inference.py's output is sparse even after many voted FPS passes
(the model was trained on single 2048-point global downsamples of the whole
scan, not dense local patches), so the raster step produces an empty
skeleton on real data. This script skips straight to what cells 12+14+22+32
of that notebook already do algorithmically -- greedy nearest-neighbor
ordering, B-spline fit, Delaunay-surface normals, tangent/normal -> quaternion
-- applied directly to the labeled 3D points instead of to raster-derived
pixel coordinates.

Output is in the SCANNER's coordinate frame, not the robot's -- the
notebook's cell 23 transformation_matrix is specific to this project's
ABB eye-to-hand calibration and is not applied here.

Known caveat carried over from the source notebook: per-triangle Delaunay
normals can flip sign between adjacent points (visible as an abrupt sign
flip in i/j/k between consecutive rows of the output). The notebook
smooths this out with smooth_normals() (3x Gaussian pass) before using the
normals for anything geometry-critical (e.g. multi-layer offsetting) --
apply the same smoothing here before using these normals for anything
beyond visualization.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.interpolate as si
from scipy.interpolate import SmoothBivariateSpline
from scipy.spatial import Delaunay, KDTree, distance_matrix
from scipy.spatial.transform import Rotation as R


def filter_outliers(bead, radius=15.0, min_neighbors=3):
    if len(bead) < min_neighbors + 1:
        return bead
    tree = KDTree(bead)
    counts = tree.query_ball_point(bead, r=radius, return_length=True)
    return bead[counts >= min_neighbors]


def order_points(points):
    def reorder(points, start):
        n = len(points)
        dist = distance_matrix(points, points)
        order = [start]
        remaining = set(range(n)) - {start}
        while remaining:
            last = order[-1]
            nxt = min(remaining, key=lambda x: dist[last, x])
            order.append(nxt)
            remaining.remove(nxt)
        return order

    def total_dist(points, order):
        dist = distance_matrix(points[order], points[order])
        return sum(dist[i, i + 1] for i in range(len(order) - 1))

    best_order, best_d = None, float("inf")
    for i in range(len(points)):
        order = reorder(points, i)
        d = total_dist(points, order)
        if d < best_d:
            best_d, best_order = d, order
    return points[best_order]


def fit_spline_path(ordered, num_samples=100, smoothing=2, degree=3):
    tck, _ = si.splprep(ordered.T, s=smoothing, k=min(degree, len(ordered) - 1))
    u = np.linspace(0, 1, num_samples)
    return np.array(si.splev(u, tck)).T


def surface_normals(work_piece, path_3D, grid_res=300, spline_smoothing=50):
    x, y, z = work_piece[:, 0], work_piece[:, 1], work_piece[:, 2]
    spline = SmoothBivariateSpline(x, y, z, s=spline_smoothing)
    X, Y = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_res), np.linspace(y.min(), y.max(), grid_res)
    )
    Z = spline.ev(X.ravel(), Y.ravel()).reshape(grid_res, grid_res)
    surf_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    tri = Delaunay(surf_points[:, :2])

    normals = []
    for point in path_3D:
        simplex = tri.find_simplex(point[:2])
        if simplex == -1:
            normals.append([0, 0, 1])
            continue
        v0, v1, v2 = surf_points[tri.simplices[simplex]]
        n = np.cross(v1 - v0, v2 - v0)
        normals.append(n / np.linalg.norm(n))
    return np.array(normals)


def path_to_quaternions(path_3D, normals):
    t_vectors = np.diff(path_3D, axis=0)
    t_vectors = np.vstack([t_vectors, t_vectors[-1]])

    quats = []
    for t_vec, n in zip(t_vectors, normals):
        t_vec = t_vec / np.linalg.norm(t_vec)
        n = n - np.dot(n, t_vec) * t_vec  # orthogonalize against the tangent (Gram-Schmidt)
        n = n / np.linalg.norm(n)
        y_axis = np.cross(n, t_vec)
        y_axis = y_axis / np.linalg.norm(y_axis)
        rot = np.vstack((t_vec, y_axis, n)).T
        q = R.from_matrix(rot).as_quat()  # [x, y, z, w]
        quats.append([q[3], q[0], q[1], q[2]])  # -> [w, x, y, z]
    return np.array(quats)


def build_trajectory(predicted_3_path, out_path, num_samples=100,
                      outlier_radius=15.0, outlier_min_neighbors=3):
    d = np.loadtxt(predicted_3_path)
    work_piece = d[d[:, 3] == 0][:, :3]
    bead_raw = d[d[:, 3] == 1][:, :3]

    bead = filter_outliers(bead_raw, radius=outlier_radius, min_neighbors=outlier_min_neighbors)
    print(f"bead points: {len(bead_raw)} -> {len(bead)} after outlier filter")
    if len(bead) < 4:
        raise ValueError(
            f"Only {len(bead)} bead points survived filtering -- too few for a spline path. "
            "Try more --passes in pointcnn_inference.py, or relax outlier_radius/min_neighbors."
        )

    ordered = order_points(bead)
    path_3D = fit_spline_path(ordered, num_samples=num_samples)
    arc_length = np.sum(np.linalg.norm(np.diff(path_3D, axis=0), axis=1))
    print(f"path_3D: {path_3D.shape}, arc length: {arc_length:.2f} mm")

    normals = surface_normals(work_piece, path_3D)
    quats = path_to_quaternions(path_3D, normals)

    df = pd.DataFrame(path_3D, columns=["x", "y", "z"])
    df = pd.concat(
        [df, pd.DataFrame(normals, columns=["i", "j", "k"]), pd.DataFrame(quats, columns=["q1", "q2", "q3", "q4"])],
        axis=1,
    ).round(4)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} pose points, still in scanner coordinates)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predicted_3_path", help="Output of pointcnn_inference.py")
    parser.add_argument("--out", default="outputs/trajectory.csv")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--outlier-radius", type=float, default=15.0)
    parser.add_argument("--outlier-min-neighbors", type=int, default=3)
    args = parser.parse_args()

    build_trajectory(
        args.predicted_3_path, args.out, num_samples=args.num_samples,
        outlier_radius=args.outlier_radius, outlier_min_neighbors=args.outlier_min_neighbors,
    )
