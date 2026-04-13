"""
SOPReferenceBank.py — stores reference features for each SOP step.

Works with both encoders:
  - DINOv2Encoder / CNNEncoder : stores np.ndarray embeddings
  - XFeatEncoder               : stores list of feature dicts

Multi-reference support:
  Each step can have MULTIPLE reference images (different angles/positions).
  Folder structure:
      data/sop_ref/
          step_01_tighten_bolt/
              ref_01.jpg
              ref_02.jpg
              ref_03.jpg       ← same step, different positions
          step_02_insert_pin/
              ref_01.jpg
              ref_02.jpg

  self.embeddings layout:
      DINOv2/CNN : list of np.ndarray  → embeddings[step_idx] = (N_refs, dim)
      XFeat      : list of lists       → embeddings[step_idx] = [feat_dict, ...]
"""

from __future__ import annotations
import os
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image


class SOPReferenceBank:

    def __init__(self, encoder):
        """
        encoder : XFeatEncoder, DINOv2Encoder, or CNNEncoder instance.
        """
        self.encoder    = encoder
        self.steps      = []     # list of step metadata dicts
        self.embeddings = []     # per-step embeddings (see module docstring)
        self._mode      = None   # 'dino' | 'xfeat' | 'lightglue'

    # ── Registration ──────────────────────────────────────────────────────────

    def register_from_root(self, root_path: str):
        """
        Load steps from a root folder where each sub-folder = one step.
        Sub-folders are sorted alphabetically (prefix with numbers to order).

        data/sop_ref/
            01_step_name/
                ref_01.jpg
                ref_02.jpg
            02_step_name/
                ref_01.jpg
        """
        root = Path(root_path)
        step_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

        if not step_dirs:
            raise ValueError(f"No sub-folders found in '{root_path}'")

        for step_idx, step_dir in enumerate(step_dirs):
            image_files = sorted([
                f for f in step_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])
            if not image_files:
                print(f"  [WARN] No images in '{step_dir.name}', skipping.")
                continue

            images = [Image.open(f).convert('RGB') for f in image_files]
            meta = {
                'step_id':   step_idx,
                'step_name': step_dir.name,
                'num_refs':  len(images),
            }
            print(f"  Step {step_idx}: '{step_dir.name}' — {len(images)} ref image(s)")
            self._encode_and_append(images, meta)

        print(f"\n  [Bank] Ready: {len(self.steps)} step(s), multi-ref per step.")

    def register_from_folder(self, folder_path: str):
        """
        Legacy single-step registration — each image = one step.
        Kept for backward compatibility.
        """
        files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not files:
            raise ValueError(f"No images found in '{folder_path}'")

        for idx, fname in enumerate(files):
            img = Image.open(os.path.join(folder_path, fname)).convert('RGB')
            meta = {
                'step_id':   idx,
                'step_name': os.path.splitext(fname)[0],
                'filename':  fname,
                'num_refs':  1,
            }
            print(f"  Registered step {idx}: {fname}")
            self._encode_and_append([img], meta)

        print(f"\n  [Bank] Ready: {len(self.steps)} step(s) (legacy single-ref mode).")

    def register_step(self, images: list, step_name: str):
        """
        Manually register one step with multiple reference images.

        images    : list of PIL.Image or np.ndarray (BGR)
        step_name : display name for this step
        """
        if not isinstance(images, list):
            images = [images]
        meta = {
            'step_id':   len(self.steps),
            'step_name': step_name,
            'num_refs':  len(images),
        }
        self._encode_and_append(images, meta)
        print(f"  [Bank] Registered step '{step_name}' with {len(images)} ref(s).")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_refs_for_step(self, step_idx: int):
        """
        Returns all reference embeddings/features for a given step.
        DINOv2/CNN : np.ndarray (N_refs, dim)
        XFeat      : list of feature dicts
        """
        return self.embeddings[step_idx]

    def num_refs(self, step_idx: int) -> int:
        """Number of reference images registered for a step."""
        refs = self.embeddings[step_idx]
        if isinstance(refs, np.ndarray):
            return refs.shape[0]
        return len(refs)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save reference bank to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if self._mode in ('xfeat', 'lightglue'):
            with open(path + '_features.pkl', 'wb') as f:
                pickle.dump(self.embeddings, f)
        else:
            # embeddings = list of np.ndarray per step → save as list
            with open(path + '_embeddings.pkl', 'wb') as f:
                pickle.dump(self.embeddings, f)

        with open(path + '_metadata.json', 'w') as f:
            json.dump({'mode': self._mode, 'steps': self.steps}, f, indent=2)

        print(f"  [Bank] Saved to '{path}' (mode={self._mode})")

    def load(self, path: str):
        """Load reference bank from disk."""
        meta_path = path + '_metadata.json'
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        with open(meta_path) as f:
            meta = json.load(f)

        self._mode = meta.get('mode', 'dino')
        self.steps = meta['steps']

        if self._mode in ('xfeat', 'lightglue'):
            pkl_path = path + '_features.pkl'
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"Feature file not found: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            pkl_path = path + '_embeddings.pkl'
            # backward compat: old .npy was single flat array
            npy_path = path + '_embeddings.npy'
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
            elif os.path.exists(npy_path):
                flat = np.load(npy_path)
                # wrap each row as a single-ref step list
                self.embeddings = [flat[i:i+1] for i in range(flat.shape[0])]
                print("  [Bank] Migrated legacy .npy → multi-ref format.")
            else:
                raise FileNotFoundError(f"Embeddings not found at '{path}'")

        print(f"  [Bank] Loaded {len(self.steps)} step(s) (mode={self._mode})")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_and_append(self, images: list, meta: dict):
        """Encode a list of images (one step) and append to self.embeddings."""
        # Auto-detect mode from first encode
        if self._mode is None:
            sample = self.encoder.encode(images[0])
            if self._is_local_feat(sample):
                enc_name = type(self.encoder).__name__.lower()
                self._mode = 'lightglue' if 'lightglue' in enc_name else 'xfeat'
            else:
                self._mode = 'dino'
            print(f"  [Bank] Encoder mode: {self._mode}")

        if self._mode in ('xfeat', 'lightglue'):
            # Each step → list of feature dicts
            feats = [self.encoder.encode(img) for img in images]
            self.embeddings.append(feats)
        else:
            # Each step → np.ndarray (N_refs, dim)
            if hasattr(self.encoder, 'encode_batch'):
                embs = self.encoder.encode_batch(images)   # (N, dim)
            else:
                embs = np.stack([self.encoder.encode(img) for img in images])
            self.embeddings.append(embs)

        self.steps.append(meta)

    @staticmethod
    def _is_local_feat(feat) -> bool:
        return isinstance(feat, dict) and 'keypoints' in feat