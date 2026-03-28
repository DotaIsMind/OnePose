"""
ONNX Runtime inference wrappers for OnePose models.

Each wrapper mimics the interface of the original PyTorch model so that
the rest of the pipeline can use them as drop-in replacements.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_session(onnx_path: str):
    """Create an ONNX Runtime InferenceSession (CPU only)."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return sess


def _simple_nms(scores: np.ndarray, nms_radius: int) -> np.ndarray:
    """Non-maximum suppression on a 2-D score map (NumPy)."""
    from scipy.ndimage import maximum_filter
    pad = nms_radius
    max_scores = maximum_filter(scores, size=2 * nms_radius + 1, mode='constant', cval=0)
    mask = scores == max_scores
    return scores * mask


def _sample_descriptors(keypoints: np.ndarray,
                        descriptors: np.ndarray,
                        s: int = 8) -> np.ndarray:
    """
    Bilinear interpolation of dense descriptors at keypoint locations.
    Matches the original SuperPoint sample_descriptors exactly.

    keypoints:   [N, 2]  (x, y) in pixel space of the FULL image
    descriptors: [C, H, W]  dense descriptor map  (H = full_H / s)
    returns:     [C, N]
    """
    C, H, W = descriptors.shape
    # Mirror the original SuperPoint normalisation:
    #   kpts = kpts - s/2 + 0.5
    #   kpts /= [(W*s - s/2 - 0.5), (H*s - s/2 - 0.5)]
    #   kpts = kpts * 2 - 1   →  in [-1, 1]
    kpts = keypoints.copy().astype(np.float32)
    kpts[:, 0] = (kpts[:, 0] - s / 2 + 0.5) / (W * s - s / 2 - 0.5) * 2 - 1
    kpts[:, 1] = (kpts[:, 1] - s / 2 + 0.5) / (H * s - s / 2 - 0.5) * 2 - 1

    # Convert normalised coords back to descriptor-map pixel coords
    # align_corners=True  →  pixel = (norm + 1) / 2 * (size - 1)
    map_x = ((kpts[:, 0] + 1) / 2 * (W - 1)).astype(np.float32)
    map_y = ((kpts[:, 1] + 1) / 2 * (H - 1)).astype(np.float32)

    # Batch remap across all channels at once using cv2.remap on each channel
    desc_sampled = np.zeros((C, len(kpts)), dtype=np.float32)
    mx = map_x.reshape(1, -1)
    my = map_y.reshape(1, -1)
    for c in range(C):
        desc_sampled[c] = cv2.remap(
            descriptors[c],
            mx, my,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).flatten()

    # L2-normalise along the channel dimension
    norms = np.linalg.norm(desc_sampled, axis=0, keepdims=True) + 1e-8
    return desc_sampled / norms


# ─────────────────────────────────────────────────────────────────────────────
# SuperPoint ONNX wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SuperPointOnnx:
    """
    Drop-in replacement for the PyTorch SuperPoint model.

    The ONNX model outputs dense score and descriptor maps; keypoint
    selection (NMS, threshold, top-k, border removal) is done in NumPy.
    """

    def __init__(self, onnx_path: str, config: dict | None = None):
        self.sess = _get_session(onnx_path)
        default_cfg = {
            'nms_radius':        4,
            'keypoint_threshold': 0.005,
            'max_keypoints':     4096,
            'remove_borders':    4,
        }
        self.cfg = {**default_cfg, **(config or {})}

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Parameters
        ----------
        image : np.ndarray  [1, 1, H, W]  float32, 0-1

        Returns
        -------
        dict with keys 'keypoints', 'scores', 'descriptors'
            keypoints:   [N, 2]  (x, y)
            scores:      [N]
            descriptors: [256, N]
        """
        if image.ndim == 3:
            image = image[np.newaxis]          # add batch dim

        scores_dense, desc_dense = self.sess.run(
            None, {'image': image.astype(np.float32)}
        )
        # scores_dense: [1, H, W]
        # desc_dense:   [1, 256, H/8, W/8]

        scores = scores_dense[0]               # [H, W]
        desc   = desc_dense[0]                 # [256, H/8, W/8]

        H, W = scores.shape

        # NMS
        scores = _simple_nms(scores, self.cfg['nms_radius'])

        # Threshold
        ys, xs = np.where(scores > self.cfg['keypoint_threshold'])
        kpt_scores = scores[ys, xs]

        # Remove borders
        b = self.cfg['remove_borders']
        mask = (ys >= b) & (ys < H - b) & (xs >= b) & (xs < W - b)
        ys, xs, kpt_scores = ys[mask], xs[mask], kpt_scores[mask]

        # Top-k
        max_kp = self.cfg['max_keypoints']
        if max_kp >= 0 and len(kpt_scores) > max_kp:
            idx = np.argsort(kpt_scores)[::-1][:max_kp]
            ys, xs, kpt_scores = ys[idx], xs[idx], kpt_scores[idx]

        # (x, y) convention
        keypoints = np.stack([xs, ys], axis=1).astype(np.float32)  # [N, 2]

        # Sample descriptors
        if len(keypoints) > 0:
            descriptors = _sample_descriptors(keypoints, desc, s=8)  # [256, N]
        else:
            descriptors = np.zeros((256, 0), dtype=np.float32)

        return {
            'keypoints':   keypoints,
            'scores':      kpt_scores,
            'descriptors': descriptors,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SuperGlue ONNX wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SuperGlueOnnx:
    """
    Drop-in replacement for the PyTorch SuperGlue model.

    The ONNX model expects pre-normalised keypoints.  This wrapper handles
    the normalisation using the image shapes passed in the data dict.
    """

    def __init__(self, onnx_path: str):
        self.sess = _get_session(onnx_path)

    @staticmethod
    def empty_match_result(num_kpts0: int, num_kpts1: int) -> '_MatchResult':
        """
        Valid match tensor shapes when one side has no keypoints. The exported
        ONNX graph cannot run kenc with N=0 or M=0 (Conv rejects empty spatial dim).
        """
        return _MatchResult(
            np.full((1, num_kpts0), -1, dtype=np.int32),
            np.full((1, num_kpts1), -1, dtype=np.int32),
            np.zeros((1, num_kpts0), dtype=np.float32),
            np.zeros((1, num_kpts1), dtype=np.float32),
        )

    @staticmethod
    def _normalize_keypoints(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
        """Normalise keypoints to roughly [-1, 1]."""
        size    = np.array([[w, h]], dtype=np.float32)
        center  = size / 2.0
        scaling = size.max() * 0.7
        return (kpts - center) / scaling

    def __call__(self, data: dict) -> dict:
        """
        Parameters
        ----------
        data : dict  (same format as original SuperGlue)
            keypoints0, keypoints1:   [1, N/M, 2]
            descriptors0, descriptors1: [1, 256, N/M]
            scores0, scores1:         [1, N/M]
            image0, image1:           [1, 1, H, W]  (used for shape only)

        Returns
        -------
        dict with matches0, matches1, matching_scores0, matching_scores1
        """
        kpts0 = data['keypoints0'][0].numpy() if hasattr(data['keypoints0'], 'numpy') \
                else np.array(data['keypoints0'][0])
        kpts1 = data['keypoints1'][0].numpy() if hasattr(data['keypoints1'], 'numpy') \
                else np.array(data['keypoints1'][0])
        n0, n1 = kpts0.shape[0], kpts1.shape[0]
        if n0 == 0 or n1 == 0:
            return self.empty_match_result(n0, n1)

        desc0 = data['descriptors0'][0].numpy() if hasattr(data['descriptors0'], 'numpy') \
                else np.array(data['descriptors0'][0])
        desc1 = data['descriptors1'][0].numpy() if hasattr(data['descriptors1'], 'numpy') \
                else np.array(data['descriptors1'][0])
        sc0   = data['scores0'][0].numpy() if hasattr(data['scores0'], 'numpy') \
                else np.array(data['scores0'][0])
        sc1   = data['scores1'][0].numpy() if hasattr(data['scores1'], 'numpy') \
                else np.array(data['scores1'][0])

        # Image shapes for normalisation
        h0, w0 = int(data['image0'].shape[2]), int(data['image0'].shape[3])
        h1, w1 = int(data['image1'].shape[2]), int(data['image1'].shape[3])

        kpts0_norm = self._normalize_keypoints(kpts0, h0, w0)[np.newaxis]  # [1,N,2]
        kpts1_norm = self._normalize_keypoints(kpts1, h1, w1)[np.newaxis]  # [1,M,2]

        feeds = {
            'kpts0':   kpts0_norm.astype(np.float32),
            'kpts1':   kpts1_norm.astype(np.float32),
            'desc0':   desc0[np.newaxis].astype(np.float32),   # [1,256,N]
            'desc1':   desc1[np.newaxis].astype(np.float32),   # [1,256,M]
            'scores0': sc0[np.newaxis].astype(np.float32),     # [1,N]
            'scores1': sc1[np.newaxis].astype(np.float32),     # [1,M]
        }

        m0, m1, ms0, ms1 = self.sess.run(None, feeds)
        # m0/m1 are float; convert back to int
        m0 = m0.astype(np.int32)
        m1 = m1.astype(np.int32)

        # Wrap in a simple object that mimics the PyTorch tensor interface
        return _MatchResult(m0, m1, ms0, ms1)


class _MatchResult:
    """Lightweight container that mimics the dict returned by SuperGlue."""

    def __init__(self, m0, m1, ms0, ms1):
        self._m0  = m0   # [1, N]
        self._m1  = m1   # [1, M]
        self._ms0 = ms0  # [1, N]
        self._ms1 = ms1  # [1, M]

    def __getitem__(self, key):
        mapping = {
            'matches0':         self._m0,
            'matches1':         self._m1,
            'matching_scores0': self._ms0,
            'matching_scores1': self._ms1,
        }
        arr = mapping[key]
        return _NumpyTensor(arr)


class _NumpyTensor:
    """Wraps a numpy array to expose .detach().cpu().numpy() interface."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __getitem__(self, idx):
        return _NumpyTensor(self._arr[idx])

    @property
    def shape(self):
        return self._arr.shape


# ─────────────────────────────────────────────────────────────────────────────
# GATsSPG ONNX wrapper
# ─────────────────────────────────────────────────────────────────────────────

class GATsSPGOnnx:
    """
    Drop-in replacement for the PyTorch GATsSuperGlue matcher.

    Accepts the same dict-based input as the original model and returns
    the same dict-based output.
    """

    def __init__(self, onnx_path: str):
        self.sess = _get_session(onnx_path)

    def __call__(self, data: dict) -> Tuple[dict, None]:
        """
        Parameters
        ----------
        data : dict
            keypoints2d:         [1, N, 2]  (torch.Tensor or np.ndarray)
            keypoints3d:         [1, M, 3]
            descriptors2d_query: [1, 256, N]
            descriptors3d_db:    [1, 256, M]
            descriptors2d_db:    [1, 256, M*num_leaf]

        Returns
        -------
        (pred_dict, None)
            pred_dict keys: matches0, matches1, matching_scores0, matching_scores1
        """
        def _to_np(x):
            if hasattr(x, 'detach'):
                return x.detach().cpu().numpy().astype(np.float32)
            return np.array(x, dtype=np.float32)

        # ONNX may omit keypoints2d/keypoints3d: GATsSPG matching uses only
        # descriptors (keypoints are unused after the empty-shape guard).
        feeds = {}
        for inp in self.sess.get_inputs():
            feeds[inp.name] = _to_np(data[inp.name])

        m0, m1, ms0, ms1 = self.sess.run(None, feeds)
        # m0/m1 are float32 from ONNX; convert to int
        m0 = m0.astype(np.int32)   # [N]
        m1 = m1.astype(np.int32)   # [M]

        pred = {
            'matches0':         _NumpyTensor(m0),
            'matches1':         _NumpyTensor(m1),
            'matching_scores0': _NumpyTensor(ms0),
            'matching_scores1': _NumpyTensor(ms1),
        }
        return pred, None