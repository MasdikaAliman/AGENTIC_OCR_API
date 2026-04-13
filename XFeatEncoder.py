"""
XFeatEncoder.py — Drop-in replacement for DINOv2Encoder using XFeat (CVPR 2024).

Key difference from DINOv2:
  - DINOv2  : global embedding  → cosine similarity (0.0–1.0)
  - XFeat   : local keypoints   → match count (0–N inliers)

The "similarity" returned by encode() is now a feature dict that
SOPReferenceBank stores. SOPVerifier calls match() to compare live vs reference.

IMPORTANT: XFeat must be installed first. See INSTALL section in this file.
"""

from __future__ import annotations
import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from icecream import ic

# ── XFeat import (from cloned repo, not pip) ──────────────────────────────────
# XFeat has no pip package. We load from the cloned accelerated_features repo.
# Set XFEAT_PATH to the folder where you cloned it, or put it next to this file.
XFEAT_REPO = os.environ.get("XFEAT_PATH", "accelerated_features")
if XFEAT_REPO not in sys.path:
    sys.path.insert(0, XFEAT_REPO)

try:
    from modules.xfeat import XFeat as _XFeatModel
except ImportError as e:
    raise ImportError(
        f"\n[XFeatEncoder] Cannot import XFeat from '{XFEAT_REPO}'.\n"
        f"Run the install steps:\n"
        f"  git clone https://github.com/verlab/accelerated_features.git\n"
        f"  pip install torch opencv-contrib-python tqdm\n"
        f"Or set XFEAT_PATH env variable to the cloned folder path.\n"
        f"Original error: {e}"
    ) from e


# ── XFeatEncoder ──────────────────────────────────────────────────────────────

class XFeatEncoder:
    """
    Drop-in replacement for DINOv2Encoder.

    Instead of a single embedding vector, this returns a feature dict:
      {
        'keypoints':   np.ndarray (N, 2)   — pixel coords
        'descriptors': np.ndarray (N, 64)  — L2-normalised descriptors
        'scores':      np.ndarray (N,)     — keypoint confidence scores
      }

    Use match() to compare a live frame against a stored reference dict.
    The match score (inlier count) replaces cosine similarity as your
    pass/fail metric. Typical good threshold: 15–40 inliers depending on
    item size and texture.

    Parameters
    ----------
    top_k : int
        Max keypoints to detect per image. 2048 is a good default for
        small crop regions. Increase to 4096 for larger regions.
    detection_threshold : float
        Minimum keypoint confidence. Lower = more keypoints but noisier.
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(self, top_k: int = 2048,
                 detection_threshold: float = 0.05,
                max_expected_inliers: int = 50,
                 device: str | None = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.max_expected_inliers = max_expected_inliers

        if device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        self.model = _XFeatModel(
            top_k=top_k,
            detection_threshold=detection_threshold,
        )
        self.top_k = top_k
        print(f"[XFeatEncoder] Loaded on {device.upper()} | top_k={top_k}")

    # ── Encode (replaces DINOv2Encoder.encode) ────────────────────────────────

    def encode(self, image) -> dict:
        """
        Extract keypoints and descriptors from a single image.

        image : PIL.Image or np.ndarray (BGR from OpenCV)

        Returns a feature dict:
          { 'keypoints': (N,2), 'descriptors': (N,64), 'scores': (N,) }
        """
        bgr = self._to_bgr(image)
        tensor = self._to_tensor(bgr)  # (1, 3, H, W)

        with torch.no_grad():
            output = self.model.detectAndCompute(tensor, top_k=self.top_k)[0]

        return {
            'keypoints':   output['keypoints'].cpu().numpy(),    # (N, 2)
            'descriptors': output['descriptors'].cpu().numpy(),  # (N, 64)
            'scores':      output['scores'].cpu().numpy(),       # (N,)
        }

    def encode_batch(self, images: list) -> list[dict]:
        """Encode a list of images. Returns a list of feature dicts."""
        return [self.encode(img) for img in images]

    # ── Match (replaces cosine similarity) ────────────────────────────────────

    def match(self, feat_ref: dict, feat_live: dict,
            ratio_thresh: float = 0.82,
            ransac: bool = True,
            ransac_thresh: float = 3.5) -> dict:

        desc_ref  = feat_ref['descriptors']
        desc_live = feat_live['descriptors']
        kpts_ref  = feat_ref['keypoints']
        kpts_live = feat_live['keypoints']

        if len(desc_ref) == 0 or len(desc_live) == 0:
            return self._empty_result()

        t_ref  = torch.from_numpy(desc_ref).float()
        t_live = torch.from_numpy(desc_live).float()

        sim_matrix = t_ref @ t_live.T  # (N, M) cosine similarities

        top2_vals, top2_idx = sim_matrix.topk(min(2, sim_matrix.shape[1]), dim=1)

        if top2_vals.shape[1] < 2:
            good_mask = torch.ones(len(desc_ref), dtype=torch.bool)
            match_idx = top2_idx[:, 0]
            match_conf = top2_vals[:, 0]  # raw cosine sim as confidence
        else:
            ratio = top2_vals[:, 0] / (top2_vals[:, 1] + 1e-8)
            good_mask  = ratio > ratio_thresh
            match_idx  = top2_idx[:, 0]
            # ── Per-match confidence: cosine sim scaled by ratio test margin ──
            # top2_vals[:, 0] is how similar the best match is (like mconf)
            match_conf = top2_vals[:, 0] * (ratio / (ratio + 1.0))  # dampened by ratio margin

        good_idx_ref  = torch.where(good_mask)[0].numpy()
        good_idx_live = match_idx[good_mask].numpy()
        mconf         = match_conf[good_mask].numpy()  # per-match confidence scores

        if len(good_idx_ref) < 4:
            return self._empty_result()

        mkpts_ref  = kpts_ref[good_idx_ref]
        mkpts_live = kpts_live[good_idx_live]
        total      = len(mkpts_ref)

        # ── RANSAC ─────────────────────────────────────────────────────────────
        inliers = total
        if ransac and total >= 4:
            _, mask = cv2.findHomography(
                mkpts_ref.reshape(-1, 1, 2),
                mkpts_live.reshape(-1, 1, 2),
                cv2.RANSAC,
                ransac_thresh,
            )
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                inliers     = int(inlier_mask.sum())
                mkpts_ref   = mkpts_ref[inlier_mask]
                mkpts_live  = mkpts_live[inlier_mask]
                # mconf       = mconf[inlier_mask]   # keep only inlier confidences

        # ── Confidence scores (analogous to LoFTR's mconf.mean()) ──────────────
        mconf_mean = float(mconf.mean()) if len(mconf) > 0 else 0.0

        geo_quality    = inliers / max(total, 1)
        abs_confidence = min(inliers / max(self.max_expected_inliers, 1), 1.0)

        # Drop-in replacement for your similarity — now includes mconf_mean
        similarity = (0.34 * geo_quality +
                    0.33 * abs_confidence +
                    0.33 * mconf_mean)

        # similarity =  mconf_mean


        ic(geo_quality, abs_confidence, mconf_mean, total)

        return {
            'inliers':       inliers,
            'total_matches': total,
            'similarity':    round(similarity, 4),
            'mconf_mean':    round(mconf_mean, 4),   # analogous to LoFTR mconf.mean()
            'mkpts_ref':     mkpts_ref,
            'mkpts_live':    mkpts_live,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_bgr(self, image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image  # already BGR from OpenCV
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        """Convert BGR np.ndarray → normalised float tensor (1,3,H,W)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)

    def _empty_result(self) -> dict:
        return {
            'inliers':       0,
            'total_matches': 0,
            'similarity':    0.0,
            'mkpts_ref':     np.empty((0, 2)),
            'mkpts_live':    np.empty((0, 2)),
        }
