from __future__ import annotations
import cv2
import numpy as np
import torch
from PIL import Image
from icecream import ic
try:
    from lightglue import LightGlue, SuperPoint, DISK, ALIKED, SIFT
    from lightglue.utils import rbd   # remove batch dimension helper
except ImportError as e:
    raise ImportError(
        "\n[LightGlueEncoder] Cannot import lightglue.\n"
        "Install it with:\n"
        "  git clone https://github.com/cvg/LightGlue.git\n"
        "  cd LightGlue && pip install -e .\n"
        f"Original error: {e}"
    ) from e

# Map config string → extractor class
_EXTRACTORS = {
    'superpoint': SuperPoint,
    'aliked':     ALIKED,
    'disk':       DISK,
    'sift':       SIFT,
}


class LightGlueEncoder:
    """
    Drop-in replacement for XFeatEncoder and DINOv2Encoder.

    encode(image)  → feature dict  { keypoints, descriptors, scores, image_size }
    match(f0, f1)  → match result  { inliers, total_matches, similarity, mkpts_ref, mkpts_live }

    The result dict is identical to XFeatEncoder.match() so SOPVerifier
    and SOPReferenceBank work without any changes.

    Parameters
    ----------
    features : str
        Backend extractor. One of: 'superpoint', 'aliked', 'disk', 'sift'.
        Recommendation for industrial inspection: 'aliked'
        Recommendation for general use: 'superpoint'
    max_num_keypoints : int
        Max keypoints per image. 1024 = fast, 2048 = balanced, 4096 = accurate.
    max_expected_inliers : int
        Used only for similarity normalisation. Set to the typical inlier
        count you see on a good match (watch console output to tune).
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    depth_confidence : float
        LightGlue early stopping per layer. -1 = disabled (max accuracy).
        0.95 = default (good speed/accuracy balance).
    width_confidence : float
        LightGlue point pruning. -1 = disabled. 0.99 = default.
    filter_threshold : float
        Minimum match confidence to keep a match. Higher = fewer but stronger.
        Default 0.1. Raise to 0.3-0.5 for stricter matching.
    """

    def __init__(
        self,
        features:            str   = 'aliked',
        max_num_keypoints:   int   = 2048,
        max_expected_inliers: int  = 50,
        device:              str | None = None,
        depth_confidence:    float = 0.95,
        width_confidence:    float = 0.99,
        filter_threshold:    float = 0.1,
    ):
        if features not in _EXTRACTORS:
            raise ValueError(
                f"Unknown features backend '{features}'. "
                f"Choose from: {list(_EXTRACTORS)}"
            )

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.features = features
        self.max_num_keypoints   = max_num_keypoints
        self.max_expected_inliers = max_expected_inliers

        print(f"[LightGlueEncoder] Loading {features.upper()} extractor "
              f"+ LightGlue on {self.device.upper()} | "
              f"max_keypoints={max_num_keypoints}")

        ExtractorClass = _EXTRACTORS[features]
        self.extractor = (
            ExtractorClass(max_num_keypoints=max_num_keypoints)
            .eval()
            .to(self.device)
        )

        self.matcher = (
            LightGlue(
                features=features,
                depth_confidence=depth_confidence,
                width_confidence=width_confidence,
                filter_threshold=filter_threshold,
            )
            .eval()
            .to(self.device)
        )

        print(f"[LightGlueEncoder] Ready.")

    # ── Encode ────────────────────────────────────────────────────────────────

    def encode(self, image) -> dict:
        """
        Extract keypoints and descriptors from a single image.

        image : PIL.Image or np.ndarray (BGR from OpenCV)

        Returns a feature dict containing:
          - All raw tensors from extractor.extract() — passed as-is to matcher
          - '_kpts_np'    np.ndarray (N,2) pixel coords — for RANSAC & display
          - '_image_size' (H, W)           — stored for reference
        """
        tensor = self._to_tensor(image)   # (3, H, W) float32 [0,1]
        H, W   = tensor.shape[1], tensor.shape[2]

        with torch.no_grad():
            # extract() returns batched dict — rbd removes batch dimension
            feats = self.extractor.extract(
                tensor.unsqueeze(0).to(self.device)   # (1,3,H,W)
            )
            feats = rbd(feats)   # → dict of (N,...) tensors, no batch dim

        # Convert keypoints to pixel space for RANSAC and display.
        # LightGlue stores keypoints normalised in [−1, 1]; convert to pixels.
        kpts_norm = feats['keypoints'].cpu().numpy()   # (N,2)
        kpts_px   = self._normalized_to_pixel(kpts_norm, H, W)

        # Build output: keep ALL extractor keys (varies by backend) +
        # add our numpy helpers prefixed with '_' (skipped by matcher).
        out = dict(feats)   # shallow copy — all torch tensors from extractor
        out['_kpts_np']    = kpts_px          # (N,2) pixel coords
        out['_image_size'] = (H, W)

        # One-time debug: log available keys so mismatches are easy to spot
        if not hasattr(self, '_keys_logged'):
            self._keys_logged = True
            tensor_keys = [k for k,v in feats.items() if isinstance(v, torch.Tensor)]
            print(f"[LightGlueEncoder] Extractor output keys: {tensor_keys}")

        return out

    def encode_batch(self, images: list) -> list[dict]:
        """Encode a list of images. Returns a list of feature dicts."""
        return [self.encode(img) for img in images]

    # ── Match ─────────────────────────────────────────────────────────────────

    def match(self, feat_ref: dict, feat_live: dict,
              ransac: bool = True,
              ransac_thresh: float = 3.5) -> dict:
        """
        Match reference features against live features using LightGlue.

        LightGlue already does its own learned filtering — RANSAC is applied
        on top for extra geometric verification.

        Returns the same dict shape as XFeatEncoder.match():
          {
            'inliers':       int
            'total_matches': int
            'similarity':    float  (0.0–1.0)
            'mkpts_ref':     np.ndarray (M,2)  pixel coords in reference
            'mkpts_live':    np.ndarray (M,2)  pixel coords in live frame
          }
        """
        # LightGlue needs tensors on the right device
        data0 = self._feat_to_device(feat_ref)
        data1 = self._feat_to_device(feat_live)

        with torch.no_grad():
            result = self.matcher({
                'image0': data0,
                'image1': data1,
            })
            result = rbd(result)   # remove batch dim

        # Get match indices
        # 'matches' shape: (M, 2) — pairs of [idx_in_ref, idx_in_live]
        # Note: some LightGlue versions use 'matches0' instead of 'matches'
        if 'matches' in result:
            matches = result['matches'].cpu().numpy()
        elif 'matches0' in result:
            # older API: matches0[i] = j means ref kpt i matches live kpt j
            m0 = result['matches0'].cpu().numpy()   # (N,) -1 = unmatched
            valid = m0 > -1
            matches = np.stack([np.where(valid)[0], m0[valid]], axis=1)
        else:
            print(f"[WARN] Unexpected matcher output keys: {list(result.keys())}")
            return self._empty_result()

        if len(matches) == 0:
            return self._empty_result()

        # Get pixel-space keypoints for matched pairs
        kpts_ref  = feat_ref['_kpts_np']
        kpts_live = feat_live['_kpts_np']

        mkpts_ref  = kpts_ref[matches[:, 0]]
        mkpts_live = kpts_live[matches[:, 1]]
        total      = len(mkpts_ref)

        # ── Optional RANSAC on top of LightGlue output ─────────────────────
        inliers = total
        if ransac and total >= 4:
            _, mask = cv2.findHomography(
                mkpts_ref.reshape(-1, 1, 2).astype(np.float32),
                mkpts_live.reshape(-1, 1, 2).astype(np.float32),
                cv2.RANSAC,
                ransac_thresh,
            )
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                inliers     = int(inlier_mask.sum())
                mkpts_ref   = mkpts_ref[inlier_mask]
                mkpts_live  = mkpts_live[inlier_mask]

        # ── Similarity score ───────────────────────────────────────────────
        geo_quality    = inliers / max(total, 1)
        abs_confidence = min(inliers / max(self.max_expected_inliers, 1), 1.0)
        similarity     = 0.5 * geo_quality + 0.5 * abs_confidence

        return {
            'inliers':       inliers,
            'total_matches': total,
            'similarity':    round(similarity, 4),
            'mkpts_ref':     mkpts_ref,
            'mkpts_live':    mkpts_live,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_tensor(self, image) -> torch.Tensor:
        """Convert PIL or BGR np.ndarray → RGB float32 tensor (3,H,W) [0,1]."""
        if isinstance(image, np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            rgb = np.array(image.convert('RGB'))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def _feat_to_device(self, feat: dict) -> dict:
        """
        Prepare a feature dict for the LightGlue matcher:
          1. Move all torch tensors to self.device
          2. Re-add the batch dimension (unsqueeze(0)) that rbd() removed
             in encode(). LightGlue.forward() expects (B, N, ...) tensors.
          3. Skip our numpy helper keys (prefixed with '_').
        """
        out = {}
        for k, v in feat.items():
            if k.startswith('_'):
                continue   # skip numpy helpers
            if isinstance(v, torch.Tensor):
                out[k] = v.unsqueeze(0).to(self.device)  # (N,...) → (1,N,...)
            else:
                out[k] = v
        return out

    @staticmethod
    def _normalized_to_pixel(kpts_norm: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        LightGlue stores keypoints normalised to [-1, 1].
        Convert back to pixel coordinates for RANSAC and display.
        """
        kpts_px = kpts_norm.copy()
        kpts_px[:, 0] = (kpts_px[:, 0] + 1) * W / 2
        kpts_px[:, 1] = (kpts_px[:, 1] + 1) * H / 2
        return kpts_px

    def _empty_result(self) -> dict:
        return {
            'inliers':       0,
            'total_matches': 0,
            'similarity':    0.0,
            'mkpts_ref':     np.empty((0, 2)),
            'mkpts_live':    np.empty((0, 2)),
        }