"""
Export OnePose models to ONNX format.

Models exported:
    1. SuperPoint  -> onnx_demo/models/superpoint.onnx
    2. SuperGlue   -> onnx_demo/models/superglue.onnx
    3. GATsSPG     -> onnx_demo/models/gatsspg.onnx

Run:
    python -m onnx_demo.export_models
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__name__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ONNX_MODEL_DIR = Path(__name__).parent / "models"
ONNX_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SuperPoint ONNX wrapper
# The original SuperPoint uses Python-level list comprehensions and dynamic
# keypoint selection that are not ONNX-traceable.  We export only the dense
# backbone (encoder + score/descriptor heads) and perform the keypoint
# selection in NumPy at runtime.
# ─────────────────────────────────────────────────────────────────────────────

class SuperPointBackbone(nn.Module):
    """
    Export the dense part of SuperPoint to ONNX.

    Inputs:
        image: [1, 1, H, W]  (float32, normalised 0-1)

    Outputs:
        scores_dense:      [1, H, W]   – per-pixel score map (after NMS removed)
        descriptors_dense: [1, 256, H//8, W//8]  – dense descriptor map
    """

    def __init__(self, superpoint_model):
        super().__init__()
        m = superpoint_model
        self.relu  = m.relu
        self.pool  = m.pool
        self.conv1a = m.conv1a; self.conv1b = m.conv1b
        self.conv2a = m.conv2a; self.conv2b = m.conv2b
        self.conv3a = m.conv3a; self.conv3b = m.conv3b
        self.conv4a = m.conv4a; self.conv4b = m.conv4b
        self.convPa = m.convPa; self.convPb = m.convPb
        self.convDa = m.convDa; self.convDb = m.convDb

    def forward(self, inp):
        x = self.relu(self.conv1a(inp))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Score head
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)                          # [1, 65, H/8, W/8]
        scores = F.softmax(scores, 1)[:, :-1]              # [1, 64, H/8, W/8]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)  # [1, H, W]

        # Descriptor head
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)                     # [1, 256, H/8, W/8]
        descriptors = F.normalize(descriptors, p=2, dim=1)

        return scores, descriptors


# ─────────────────────────────────────────────────────────────────────────────
# SuperGlue ONNX wrapper
# SuperGlue's forward uses image shapes for keypoint normalisation.
# We fold the normalisation into the wrapper so the ONNX graph only needs
# keypoints, descriptors, and scores as inputs.
# ─────────────────────────────────────────────────────────────────────────────

class SuperGlueOnnxWrapper(nn.Module):
    """
    SuperGlue wrapper that accepts pre-normalised keypoints.

    Inputs  (all float32):
        kpts0:   [1, N, 2]   normalised keypoints image-0
        kpts1:   [1, M, 2]   normalised keypoints image-1
        desc0:   [1, 256, N] descriptors image-0
        desc1:   [1, 256, M] descriptors image-1
        scores0: [1, N]      keypoint scores image-0
        scores1: [1, M]      keypoint scores image-1

    Outputs (float32):
        matches0:        [1, N]  index into kpts1 (-1 = unmatched)
        matches1:        [1, M]  index into kpts0 (-1 = unmatched)
        matching_scores0:[1, N]
        matching_scores1:[1, M]
    """


    def __init__(self, superglue_model, sinkhorn_iterations: int = 20):
        super().__init__()
        self.kenc      = superglue_model.kenc
        self.gnn       = superglue_model.gnn
        self.final_proj = superglue_model.final_proj
        self.bin_score  = superglue_model.bin_score

        # Use fewer Sinkhorn iterations for ONNX export to keep graph size
        # manageable; 20 iterations is sufficient for good matching quality.
        self.sinkhorn_iterations = sinkhorn_iterations
        self.match_threshold     = superglue_model.config['match_threshold']

    def forward(self, kpts0, kpts1, desc0, desc1, scores0, scores1):
        # Keypoint MLP encoder
        desc0 = desc0 + self.kenc(kpts0, scores0)
        desc1 = desc1 + self.kenc(kpts1, scores1)

        # Multi-layer Transformer
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final projection
        mdesc0 = self.final_proj(desc0)
        mdesc1 = self.final_proj(desc1)

        # Matching scores
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / (mdesc0.shape[1] ** 0.5)

        # Optimal transport (Sinkhorn)
        scores = self._log_optimal_transport(scores, self.bin_score,
                                             self.sinkhorn_iterations)

        max0 = scores[:, :-1, :-1].max(2)
        max1 = scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices

        # arange_like helpers – must be traceable
        arange0 = torch.arange(indices0.shape[1], device=indices0.device)[None]
        arange1 = torch.arange(indices1.shape[1], device=indices1.device)[None]

        mutual0 = arange0 == indices1.gather(1, indices0)
        mutual1 = arange1 == indices0.gather(1, indices1)

        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.match_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return indices0.float(), indices1.float(), mscores0, mscores1

    @staticmethod
    def _log_optimal_transport(scores, alpha, iters):
        b, m, n = scores.shape
        one  = scores.new_tensor(1)
        ms   = (m * one).to(scores)
        ns   = (n * one).to(scores)

        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha_exp = alpha.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, alpha_exp], -1)], 1)

        norm   = -(ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu = log_mu[None].expand(b, -1)
        log_nu = log_nu[None].expand(b, -1)

        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(couplings + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(couplings + u.unsqueeze(2), dim=1)
        Z = couplings + u.unsqueeze(2) + v.unsqueeze(1) - norm
        return Z


# ─────────────────────────────────────────────────────────────────────────────
# GATsSPG ONNX wrapper
# ─────────────────────────────────────────────────────────────────────────────

class GATsSPGOnnxWrapper(nn.Module):
    """
    Thin wrapper around GATsSuperGlue.matcher so we can pass tensors
    instead of a dict (required for ONNX tracing).

    Inputs (all float32):
        keypoints2d:        [1, N, 2]
        keypoints3d:        [1, M, 3]
        descriptors2d_query:[1, 256, N]
        descriptors3d_db:   [1, 256, M]
        descriptors2d_db:   [1, 256, M*num_leaf]

    Outputs (float32):
        matches0:        [N]
        matches1:        [M]
        matching_scores0:[N]
        matching_scores1:[M]
    """

    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher

    def forward(self, keypoints2d, keypoints3d,
                descriptors2d_query, descriptors3d_db, descriptors2d_db):
        data = {
            'keypoints2d':         keypoints2d,
            'keypoints3d':         keypoints3d,
            'descriptors2d_query': descriptors2d_query,
            'descriptors3d_db':    descriptors3d_db,
            'descriptors2d_db':    descriptors2d_db,
        }
        pred, _ = self.matcher(data)
        return (pred['matches0'].float(),
                pred['matches1'].float(),
                pred['matching_scores0'],
                pred['matching_scores1'])


# ─────────────────────────────────────────────────────────────────────────────
# Export functions
# ─────────────────────────────────────────────────────────────────────────────

def export_superpoint(model_path: str, output_path: str):
    """Export SuperPoint backbone to ONNX."""
    print(f"\n[1/3] Exporting SuperPoint  ->  {output_path}")
    from src.models.extractors.SuperPoint.superpoint import SuperPoint
    from onnx_demo.sfm.extract_features import confs
    from onnx_demo.utils.model_io import load_network

    conf = confs['superpoint']['conf']
    sp = SuperPoint(conf)
    sp.eval()
    load_network(sp, model_path, force=True)

    wrapper = SuperPointBackbone(sp)
    wrapper.eval()

    dummy = torch.zeros(1, 1, 512, 512)

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=['image'],
        output_names=['scores_dense', 'descriptors_dense'],
        dynamic_axes={
            'image':             {2: 'height', 3: 'width'},
            'scores_dense':      {1: 'height', 2: 'width'},
            'descriptors_dense': {2: 'h8',     3: 'w8'},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    import onnx
    onnx.checker.check_model(onnx.load(output_path))
    print(f"    ✓ SuperPoint exported and verified.")


def export_superglue(model_path: str, output_path: str):
    """Export SuperGlue to ONNX."""
    print(f"\n[2/3] Exporting SuperGlue   ->  {output_path}")
    from src.models.matchers.SuperGlue.superglue import SuperGlue
    from onnx_demo.sfm.match_features import confs
    from onnx_demo.utils.model_io import load_network

    conf = confs['superglue']['conf']
    sg = SuperGlue(conf)
    sg.eval()
    load_network(sg, model_path, force=True)

    wrapper = SuperGlueOnnxWrapper(sg)
    wrapper.eval()

    N, M = 256, 256
    dummy_kpts0   = torch.zeros(1, N, 2)
    dummy_kpts1   = torch.zeros(1, M, 2)
    dummy_desc0   = torch.zeros(1, 256, N)
    dummy_desc1   = torch.zeros(1, 256, M)
    dummy_scores0 = torch.zeros(1, N)
    dummy_scores1 = torch.zeros(1, M)

    torch.onnx.export(
        wrapper,
        (dummy_kpts0, dummy_kpts1, dummy_desc0, dummy_desc1,
         dummy_scores0, dummy_scores1),
        output_path,
        input_names=['kpts0', 'kpts1', 'desc0', 'desc1', 'scores0', 'scores1'],
        output_names=['matches0', 'matches1', 'matching_scores0', 'matching_scores1'],
        dynamic_axes={
            'kpts0':            {1: 'N'},
            'kpts1':            {1: 'M'},
            'desc0':            {2: 'N'},
            'desc1':            {2: 'M'},
            'scores0':          {1: 'N'},
            'scores1':          {1: 'M'},
            'matches0':         {1: 'N'},
            'matches1':         {1: 'M'},
            'matching_scores0': {1: 'N'},
            'matching_scores1': {1: 'M'},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    import onnx
    onnx.checker.check_model(onnx.load(output_path))
    print(f"    ✓ SuperGlue exported and verified.")


def export_gatsspg(model_path: str, output_path: str):
    """Export GATsSPG matcher to ONNX."""
    print(f"\n[3/3] Exporting GATsSPG     ->  {output_path}")
    from src.models.GATsSPG_lightning_model import LitModelGATsSPG

    lit = LitModelGATsSPG.load_from_checkpoint(model_path, map_location='cpu')
    lit.eval()

    wrapper = GATsSPGOnnxWrapper(lit.matcher)
    wrapper.eval()

    N, M, num_leaf = 200, 100, 8
    dummy_kpts2d   = torch.zeros(1, N, 2)
    dummy_kpts3d   = torch.zeros(1, M, 3)
    dummy_desc2d_q = torch.zeros(1, 256, N)
    dummy_desc3d_db= torch.zeros(1, 256, M)
    dummy_desc2d_db= torch.zeros(1, 256, M * num_leaf)

    torch.onnx.export(
        wrapper,
        (dummy_kpts2d, dummy_kpts3d, dummy_desc2d_q,
         dummy_desc3d_db, dummy_desc2d_db),
        output_path,
        input_names=[
            'keypoints2d', 'keypoints3d',
            'descriptors2d_query', 'descriptors3d_db', 'descriptors2d_db',
        ],
        output_names=[
            'matches0', 'matches1',
            'matching_scores0', 'matching_scores1',
        ],
        dynamic_axes={
            'keypoints2d':         {1: 'N'},
            'keypoints3d':         {1: 'M'},
            'descriptors2d_query': {2: 'N'},
            'descriptors3d_db':    {2: 'M'},
            'descriptors2d_db':    {2: 'M_leaf'},
            'matches0':            {0: 'N'},
            'matches1':            {0: 'M'},
            'matching_scores0':    {0: 'N'},
            'matching_scores1':    {0: 'M'},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    import onnx
    onnx.checker.check_model(onnx.load(output_path))
    print(f"    ✓ GATsSPG exported and verified.")


def main():
    print("=" * 60)
    print("  OnePose → ONNX Model Export")
    print("=" * 60)

    sp_src  = str(PROJECT_ROOT / "data/models/extractors/SuperPoint/superpoint_v1.pth")
    sg_src  = str(PROJECT_ROOT / "data/models/matchers/SuperGlue/superglue_outdoor.pth")
    gat_src = str(PROJECT_ROOT / "data/models/checkpoints/onepose/GATsSPG.ckpt")

    sp_dst  = str(ONNX_MODEL_DIR / "superpoint.onnx")
    sg_dst  = str(ONNX_MODEL_DIR / "superglue.onnx")
    gat_dst = str(ONNX_MODEL_DIR / "gatsspg.onnx")

    export_superpoint(sp_src,  sp_dst)
    export_superglue (sg_src,  sg_dst)
    export_gatsspg   (gat_src, gat_dst)

    print("\n" + "=" * 60)
    print("  All models exported successfully!")
    print(f"  Output directory: {ONNX_MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()