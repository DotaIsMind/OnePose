#!/usr/bin/env python3
"""
单文件 SfM 预处理流水线，对应 ``run.py`` 中 ``type: sfm`` 的路径。

- 图像枚举、降采样、COLMAP 三角化、后处理与 ``run.py`` 一致。
- 特征提取 / 匹配：优先 ONNX Runtime（CPU）；否则在启用 ``--backend torch_cpu`` 时用 PyTorch
  CPU，并对 ``.cuda()`` 做 no-op 补丁（原 ``extract_features`` / ``match_features`` 硬编码 CUDA）。
- ``filter_by_3d_box``：本文件内 NumPy 实现，不依赖 PyTorch。

用法示例::

    cd /path/to/OnePose
    python run_single.py \\
        --data-dir \"$(pwd)/data/demo/test_coffee test_coffee-annotate\" \\
        --outputs-dir \"$(pwd)/data/demo/test_coffee/sfm_model\"

可选 ONNX（推荐无 GPU 且不想装 PyTorch CUDA）::

    export ONEPOSE_SUPERPOINT_ONNX=/path/to/superpoint.onnx
    export ONEPOSE_SUPERGLUE_ONNX=/path/to/superglue.onnx
    python run_single.py --backend onnx ...
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import os.path as osp
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# NumPy：3D 包围盒过滤（替代 src/sfm/postprocess/filter_points.filter_by_3d_box 中的 torch）
# --------------------------------------------------------------------------- #


def _filter_by_track_length(points3D, track_length):
    from src.utils.colmap import read_write_model

    idxs_3d = sorted(points3D.keys())
    xyzs = np.empty((0, 3), dtype=np.float64)
    points_idxs = np.empty((0,), dtype=np.int64)
    for idx_3d in idxs_3d:
        if len(points3D[idx_3d].point2D_idxs) < track_length:
            continue
        xyz = points3D[idx_3d].xyz.reshape(1, -1)
        xyzs = np.append(xyzs, xyz, axis=0)
        points_idxs = np.append(points_idxs, idx_3d)
    return xyzs.astype(np.float32), points_idxs


def _filter_by_3d_box_np(points: np.ndarray, points_idxs: np.ndarray, box_path: str):
    corner_in_cano = np.loadtxt(box_path).astype(np.float32)
    assert points.shape[1] == 3
    p = points.astype(np.float32)

    def filter_(bbox_3d, pts):
        v45 = bbox_3d[5] - bbox_3d[4]
        v40 = bbox_3d[0] - bbox_3d[4]
        v47 = bbox_3d[7] - bbox_3d[4]
        pts_c = pts - bbox_3d[4]
        m0 = pts_c @ v45
        m1 = pts_c @ v40
        m2 = pts_c @ v47
        cs = []
        for m, v in zip([m0, m1, m2], [v45, v40, v47]):
            vv = float(v @ v)
            c = (m > 0) & (m < vv)
            cs.append(c)
        cs = cs[0] & cs[1] & cs[2]
        passed = np.nonzero(cs)[0]
        return passed

    passed_inds = filter_(corner_in_cano, p)
    filtered_xyzs = p[passed_inds]
    passed_idxs = points_idxs[passed_inds]
    return filtered_xyzs, passed_idxs


def _filter_3d_np(model_path: str, track_length: int, box_path: str):
    from src.utils.colmap import read_write_model

    points_model_path = osp.join(model_path, "points3D.bin")
    points3D = read_write_model.read_points3d_binary(points_model_path)
    xyzs, points_idxs = _filter_by_track_length(points3D, track_length)
    return _filter_by_3d_box_np(xyzs, points_idxs, box_path)


def _merge_points(xyzs, points_idxs, dist_threshold=1e-3):
    """与 ``filter_points.merge`` 相同逻辑，避免为合并单独 import 含 torch 的模块。"""
    from scipy.spatial.distance import pdist, squareform

    if not isinstance(xyzs, np.ndarray):
        xyzs = np.array(xyzs)
    dist = pdist(xyzs, "euclidean")
    distance_matrix = squareform(dist)
    close_than_thresh = distance_matrix < dist_threshold

    ret_points_count = 0
    ret_points = np.empty(shape=[0, 3])
    ret_idxs = {}
    points3D_idx_record = []
    for j in range(distance_matrix.shape[0]):
        idxs = close_than_thresh[j]
        if np.isin(points_idxs[idxs], points3D_idx_record).any():
            continue
        points = np.mean(xyzs[idxs], axis=0)
        ret_points = np.append(ret_points, points.reshape(1, 3), axis=0)
        ret_idxs[ret_points_count] = points_idxs[idxs]
        ret_points_count += 1
        points3D_idx_record = points3D_idx_record + points_idxs[idxs].tolist()
    return ret_points, ret_idxs


# --------------------------------------------------------------------------- #
# ONNX：SuperPoint / SuperGlue（CPU，与 test_onnx/onnx_demo/pipeline_single 思路一致）
# --------------------------------------------------------------------------- #


def _ort_session(onnx_path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(
        onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )


def _simple_nms(scores: np.ndarray, nms_radius: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter

    max_scores = maximum_filter(scores, size=2 * nms_radius + 1, mode="constant", cval=0)
    mask = scores == max_scores
    return scores * mask


def _sample_descriptors(keypoints: np.ndarray, descriptors: np.ndarray, s: int = 8) -> np.ndarray:
    C, H, W = descriptors.shape
    kpts = keypoints.copy().astype(np.float32)
    kpts[:, 0] = (kpts[:, 0] - s / 2 + 0.5) / (W * s - s / 2 - 0.5) * 2 - 1
    kpts[:, 1] = (kpts[:, 1] - s / 2 + 0.5) / (H * s - s / 2 - 0.5) * 2 - 1
    map_x = ((kpts[:, 0] + 1) / 2 * (W - 1)).astype(np.float32).reshape(1, -1)
    map_y = ((kpts[:, 1] + 1) / 2 * (H - 1)).astype(np.float32).reshape(1, -1)
    import cv2

    desc_sampled = np.zeros((C, len(kpts)), dtype=np.float32)
    for c in range(C):
        desc_sampled[c] = cv2.remap(
            descriptors[c],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).flatten()
    norms = np.linalg.norm(desc_sampled, axis=0, keepdims=True) + 1e-8
    return desc_sampled / norms


class _SuperPointOnnx:
    def __init__(self, onnx_path: str, sp_conf: dict | None = None):
        self.sess = _ort_session(onnx_path)
        default_cfg = {
            "nms_radius": 3,
            "keypoint_threshold": 0.005,
            "max_keypoints": 4096,
            "remove_borders": 4,
        }
        self.cfg = {**default_cfg, **(sp_conf or {})}

    def __call__(self, image_chw: np.ndarray) -> dict:
        """image_chw: (1, H, W) float32 in [0,1]."""
        if image_chw.ndim == 3:
            batch = image_chw[np.newaxis]
        else:
            batch = image_chw
        scores_dense, desc_dense = self.sess.run(None, {"image": batch.astype(np.float32)})
        scores = scores_dense[0]
        desc = desc_dense[0]
        H, W = scores.shape
        scores = _simple_nms(scores, self.cfg["nms_radius"])
        ys, xs = np.where(scores > self.cfg["keypoint_threshold"])
        kpt_scores = scores[ys, xs]
        b = self.cfg["remove_borders"]
        mask = (ys >= b) & (ys < H - b) & (xs >= b) & (xs < W - b)
        ys, xs, kpt_scores = ys[mask], xs[mask], kpt_scores[mask]
        max_kp = self.cfg["max_keypoints"]
        if max_kp >= 0 and len(kpt_scores) > max_kp:
            idx = np.argsort(kpt_scores)[::-1][:max_kp]
            ys, xs, kpt_scores = ys[idx], xs[idx], kpt_scores[idx]
        keypoints = np.stack([xs, ys], axis=1).astype(np.float32)
        if len(keypoints) > 0:
            descriptors = _sample_descriptors(keypoints, desc, s=8)
        else:
            descriptors = np.zeros((256, 0), dtype=np.float32)
        return {
            "keypoints": keypoints,
            "scores": kpt_scores,
            "descriptors": descriptors,
        }


def _names_to_pair(name0, name1):
    return "_".join((name0.replace("/", "-"), name1.replace("/", "-")))


def _normalize_keypoints_sg(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    size = np.array([[w, h]], dtype=np.float32)
    center = size / 2.0
    scaling = size.max() * 0.7
    return (kpts - center) / scaling


class _SuperGlueOnnx:
    def __init__(self, onnx_path: str):
        self.sess = _ort_session(onnx_path)

    def match_pair(self, feats0: dict, feats1: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kpts0 = feats0["keypoints"].astype(np.float32)
        kpts1 = feats1["keypoints"].astype(np.float32)
        desc0 = feats0["descriptors"].astype(np.float32)
        desc1 = feats1["descriptors"].astype(np.float32)
        sc0 = feats0["scores"].astype(np.float32)
        sc1 = feats1["scores"].astype(np.float32)
        h0, w0 = int(feats0["image_size"][0]), int(feats0["image_size"][1])
        h1, w1 = int(feats1["image_size"][0]), int(feats1["image_size"][1])
        kpts0_n = _normalize_keypoints_sg(kpts0, h0, w0)[np.newaxis]
        kpts1_n = _normalize_keypoints_sg(kpts1, h1, w1)[np.newaxis]
        feeds = {
            "kpts0": kpts0_n.astype(np.float32),
            "kpts1": kpts1_n.astype(np.float32),
            "desc0": desc0[np.newaxis].astype(np.float32),
            "desc1": desc1[np.newaxis].astype(np.float32),
            "scores0": sc0[np.newaxis].astype(np.float32),
            "scores1": sc1[np.newaxis].astype(np.float32),
        }
        m0, m1, ms0, ms1 = self.sess.run(None, feeds)
        return m0.astype(np.int32), m1.astype(np.int32), ms0, ms1


def _extract_features_onnx(img_lists: list, feature_out: str, cfg):
    import cv2

    sp_onnx = os.environ.get(
        "ONEPOSE_SUPERPOINT_ONNX",
        str(ROOT / "data/models/onnx/superpoint_v1.onnx"),
    )
    if not osp.isfile(sp_onnx):
        raise FileNotFoundError(
            f"未找到 SuperPoint ONNX: {sp_onnx}\n"
            "请设置 ONEPOSE_SUPERPOINT_ONNX 或放置模型到默认路径，或使用 --backend torch_cpu。"
        )
    model = _SuperPointOnnx(sp_onnx)
    feature_file = h5py.File(feature_out, "w")
    logging.info("Exporting features (ONNX) to %s", feature_out)
    for img_path in tqdm.tqdm(img_lists):
        mode = cv2.IMREAD_GRAYSCALE
        image = cv2.imread(img_path, mode)
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")
        size = np.array(image.shape[:2], dtype=np.int64)
        image_f = (image.astype(np.float32) / 255.0)[None]
        pred = model(image_f)
        pred["image_size"] = size
        grp = feature_file.create_group(str(img_path))
        for k, v in pred.items():
            grp.create_dataset(k, data=v)
    feature_file.close()
    logging.info("Finished exporting features (ONNX).")


def _match_features_onnx(cfg, feature_path: str, covis_pairs: str, matches_out: str):
    sg_onnx = os.environ.get(
        "ONEPOSE_SUPERGLUE_ONNX",
        str(ROOT / "data/models/onnx/superglue_outdoor.onnx"),
    )
    if not osp.isfile(sg_onnx):
        raise FileNotFoundError(
            f"未找到 SuperGlue ONNX: {sg_onnx}\n"
            "请设置 ONEPOSE_SUPERGLUE_ONNX 或使用 --backend torch_cpu。"
        )
    matcher = _SuperGlueOnnx(sg_onnx)
    feature_file = h5py.File(feature_path, "r")
    with open(covis_pairs, "r") as f:
        pair_list = f.read().rstrip("\n").split("\n")
    match_file = h5py.File(matches_out, "w")
    matched = set()
    logging.info("Exporting matches (ONNX) to %s", matches_out)
    for pair in tqdm.tqdm(pair_list):
        if not pair.strip():
            continue
        name0, name1 = pair.split(" ")
        pkey = _names_to_pair(name0, name1)
        if len({(name0, name1), (name1, name0)} & matched) or pkey in match_file:
            continue
        f0 = feature_file[name0]
        f1 = feature_file[name1]
        feats0 = {k: f0[k][()] for k in f0.keys()}
        feats1 = {k: f1[k][()] for k in f1.keys()}
        m0, m1, ms0, ms1 = matcher.match_pair(feats0, feats1)
        grp = match_file.create_group(pkey)
        grp.create_dataset("matches0", data=m0[0].astype(np.int32))
        grp.create_dataset("matches1", data=m1[0].astype(np.int32))
        grp.create_dataset("matching_scores0", data=ms0[0].astype(np.float16))
        grp.create_dataset("matching_scores1", data=ms1[0].astype(np.float16))
        matched |= {(name0, name1), (name1, name0)}
    match_file.close()
    feature_file.close()
    logging.info("Finished exporting matches (ONNX).")


# --------------------------------------------------------------------------- #
# PyTorch CPU：对 .cuda() 打补丁后调用原有 main
# --------------------------------------------------------------------------- #


def _apply_torch_cpu_cuda_patch():
    import torch

    def _noop_cuda(self, *a, **k):
        return self

    torch.Tensor.cuda = _noop_cuda  # type: ignore[method-assign]
    torch.nn.Module.cuda = _noop_cuda  # type: ignore[method-assign]


def _extract_match_torch_cpu(cfg, img_lists, outputs_dir, feature_out, covis_pairs_out, matches_out):
    from src.sfm import extract_features, match_features, pairs_from_poses, generate_empty, triangulation

    _apply_torch_cpu_cuda_patch()
    extract_features.main(img_lists, feature_out, cfg)
    pairs_from_poses.covis_from_pose(
        img_lists, covis_pairs_out, cfg.sfm.covis_num, max_rotation=cfg.sfm.rotation_thresh
    )
    match_features.main(cfg, feature_out, covis_pairs_out, matches_out, vis_match=False)
    empty_dir = osp.join(outputs_dir, "sfm_empty")
    deep_sfm_dir = osp.join(outputs_dir, "sfm_ws")
    generate_empty.generate_model(img_lists, empty_dir)
    triangulation.main(
        deep_sfm_dir,
        empty_dir,
        outputs_dir,
        covis_pairs_out,
        feature_out,
        matches_out,
        image_dir=None,
    )


def _extract_match_onnx(cfg, img_lists, outputs_dir, feature_out, covis_pairs_out, matches_out):
    from src.sfm import pairs_from_poses, generate_empty, triangulation

    _extract_features_onnx(img_lists, feature_out, cfg)
    pairs_from_poses.covis_from_pose(
        img_lists, covis_pairs_out, cfg.sfm.covis_num, max_rotation=cfg.sfm.rotation_thresh
    )
    _match_features_onnx(cfg, feature_out, covis_pairs_out, matches_out)
    empty_dir = osp.join(outputs_dir, "sfm_empty")
    deep_sfm_dir = osp.join(outputs_dir, "sfm_ws")
    generate_empty.generate_model(img_lists, empty_dir)
    triangulation.main(
        deep_sfm_dir,
        empty_dir,
        outputs_dir,
        covis_pairs_out,
        feature_out,
        matches_out,
        image_dir=None,
    )


# --------------------------------------------------------------------------- #
# 与 run.py 对齐的编排
# --------------------------------------------------------------------------- #


def _build_cfg(
    work_dir: Path,
    max_num_kp3d: int,
    max_num_kp2d: int,
    data_dir_str: str,
    outputs_dir_fmt: str,
    detection_model_path: str,
    matching_model_path: str,
    down_ratio: int,
    covis_num: int,
    rotation_thresh: float,
    redo: bool,
):
    network = SimpleNamespace(
        detection="superpoint",
        detection_model_path=detection_model_path,
        matching="superglue",
        matching_model_path=matching_model_path,
    )
    dataset = SimpleNamespace(
        max_num_kp3d=max_num_kp3d,
        max_num_kp2d=max_num_kp2d,
        data_dir=data_dir_str,
        outputs_dir=outputs_dir_fmt,
    )
    sfm = SimpleNamespace(
        down_ratio=down_ratio,
        covis_num=covis_num,
        rotation_thresh=rotation_thresh,
    )
    return SimpleNamespace(
        type="sfm",
        work_dir=str(work_dir),
        redo=redo,
        network=network,
        dataset=dataset,
        sfm=sfm,
    )


def sfm_run(cfg, backend: str):
    data_dirs = cfg.dataset.data_dir
    down_ratio = cfg.sfm.down_ratio
    data_dirs = [data_dirs] if isinstance(data_dirs, str) else data_dirs

    for data_dir in data_dirs:
        logging.info("Processing %s", data_dir)
        root_dir = data_dir.split(" ")[0]
        sub_dirs = data_dir.split(" ")[1:]

        img_lists = []
        for sub_dir in sub_dirs:
            seq_dir = osp.join(root_dir, sub_dir)
            img_lists += glob.glob(str(Path(seq_dir)) + "/color/*.png", recursive=True)

        down_img_lists = []
        for img_file in img_lists:
            index = int(img_file.split("/")[-1].split(".")[0])
            if index % down_ratio == 0:
                down_img_lists.append(img_file)
        img_lists = down_img_lists

        if len(img_lists) == 0:
            logging.info("No png image in %s", root_dir)
            continue

        obj_name = root_dir.split("/")[-1]
        outputs_dir_root = cfg.dataset.outputs_dir.format(obj_name)

        sfm_core_run(cfg, img_lists, outputs_dir_root, backend)
        postprocess_run(cfg, img_lists, root_dir, outputs_dir_root)


def sfm_core_run(cfg, img_lists: list, outputs_dir_root: str, backend: str):
    outputs_dir = osp.join(
        outputs_dir_root,
        "outputs" + "_" + cfg.network.detection + "_" + cfg.network.matching,
    )
    feature_out = osp.join(outputs_dir, f"feats-{cfg.network.detection}.h5")
    covis_pairs_out = osp.join(outputs_dir, f"pairs-covis{cfg.sfm.covis_num}.txt")
    matches_out = osp.join(outputs_dir, f"matches-{cfg.network.matching}.h5")

    if cfg.redo:
        if osp.exists(outputs_dir):
            import shutil

            shutil.rmtree(outputs_dir)
        Path(outputs_dir).mkdir(exist_ok=True, parents=True)

        be = backend
        if be == "auto":
            sp_def = os.environ.get(
                "ONEPOSE_SUPERPOINT_ONNX",
                str(ROOT / "data/models/onnx/superpoint_v1.onnx"),
            )
            sg_def = os.environ.get(
                "ONEPOSE_SUPERGLUE_ONNX",
                str(ROOT / "data/models/onnx/superglue_outdoor.onnx"),
            )
            be = "onnx" if (osp.isfile(sp_def) and osp.isfile(sg_def)) else "torch_cpu"
            logging.info("auto backend -> %s", be)

        if be == "onnx":
            _extract_match_onnx(cfg, img_lists, outputs_dir, feature_out, covis_pairs_out, matches_out)
        elif be == "torch_cpu":
            _extract_match_torch_cpu(cfg, img_lists, outputs_dir, feature_out, covis_pairs_out, matches_out)
        else:
            raise ValueError(backend)


def postprocess_run(cfg, img_lists: list, root_dir: str, outputs_dir_root: str):
    from src.sfm.postprocess import feature_process, filter_tkl

    bbox_path = osp.join(root_dir, "box3d_corners.txt")
    outputs_dir = osp.join(
        outputs_dir_root,
        "outputs" + "_" + cfg.network.detection + "_" + cfg.network.matching,
    )
    feature_out = osp.join(outputs_dir, f"feats-{cfg.network.detection}.h5")
    deep_sfm_dir = osp.join(outputs_dir, "sfm_ws")
    model_path = osp.join(deep_sfm_dir, "model")

    track_length, points_count_list = filter_tkl.get_tkl(model_path, thres=cfg.dataset.max_num_kp3d, show=False)
    filter_tkl.vis_tkl_filtered_pcds(model_path, points_count_list, track_length, outputs_dir)

    xyzs, points_idxs = _filter_3d_np(model_path, track_length, bbox_path)
    merge_xyzs, merge_idxs = _merge_points(xyzs, points_idxs, dist_threshold=1e-3)
    feature_process.get_kpt_ann(cfg, img_lists, feature_out, outputs_dir, merge_idxs, merge_xyzs)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="OnePose SfM 单文件流水线 (CPU)")
    p.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT,
        help="项目根目录（默认：本文件所在目录）",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help='例如: "$PWD/data/demo/test_coffee test_coffee-annotate"',
    )
    p.add_argument(
        "--outputs-dir",
        type=str,
        required=True,
        help='例如: "$PWD/data/demo/test_coffee/sfm_model"（内部会 format 对象名）',
    )
    p.add_argument("--max-num-kp3d", type=int, default=2500)
    p.add_argument("--max-num-kp2d", type=int, default=1000)
    p.add_argument(
        "--detection-model",
        type=str,
        default=None,
        help="SuperPoint .pth（torch_cpu 后端需要）",
    )
    p.add_argument(
        "--matching-model",
        type=str,
        default=None,
        help="SuperGlue .pth（torch_cpu 后端需要）",
    )
    p.add_argument("--down-ratio", type=int, default=5)
    p.add_argument("--covis-num", type=int, default=10)
    p.add_argument("--rotation-thresh", type=float, default=50.0)
    p.add_argument("--redo", action="store_true", default=True)
    p.add_argument("--no-redo", action="store_false", dest="redo")
    p.add_argument(
        "--backend",
        choices=["auto", "onnx", "torch_cpu"],
        default="auto",
        help="特征后端：auto 在存在默认 ONNX 路径时用 onnx，否则 torch_cpu",
    )
    args = p.parse_args()
    wd = args.work_dir.resolve()
    det = args.detection_model or str(wd / "data/models/extractors/SuperPoint/superpoint_v1.pth")
    mat = args.matching_model or str(wd / "data/models/matchers/SuperGlue/superglue_outdoor.pth")

    cfg = _build_cfg(
        work_dir=wd,
        max_num_kp3d=args.max_num_kp3d,
        max_num_kp2d=args.max_num_kp2d,
        data_dir_str=args.data_dir,
        outputs_dir_fmt=args.outputs_dir,
        detection_model_path=det,
        matching_model_path=mat,
        down_ratio=args.down_ratio,
        covis_num=args.covis_num,
        rotation_thresh=args.rotation_thresh,
        redo=args.redo,
    )
    sfm_run(cfg, backend=args.backend)


if __name__ == "__main__":
    main()
