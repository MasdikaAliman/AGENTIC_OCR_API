"""
SOPVerifier.py — verifies a live frame against SOP reference images.

Works with both encoders:
  - DINOv2Encoder / CNNEncoder : compares cosine similarity of global embeddings
  - XFeatEncoder               : compares inlier count of local feature matches

Multi-reference support:
  Each step can have multiple reference images. The verifier matches the live
  frame against ALL refs for each step and picks the highest scoring one.

Pass threshold meaning per encoder:
  DINOv2/CNN : cosine similarity  0.0–1.0  (e.g. 0.82)
  XFeat      : similarity score   0.0–1.0  (combined inlier+conf score)
"""

from __future__ import annotations
import numpy as np
from SOPReferenceBank import SOPReferenceBank
from icecream import ic


class SOPVerifier:

    def __init__(self, encoder, bank: SOPReferenceBank,
                 pass_threshold: float = 0.4):
        """
        encoder        : XFeatEncoder, DINOv2Encoder, or CNNEncoder
        bank           : SOPReferenceBank (already loaded/registered)
        pass_threshold : similarity threshold to consider a step matched
                         XFeat      → 0.3–0.6 (combined similarity score)
                         DINOv2/CNN → 0.75–0.90 (cosine similarity)
        """
        self.encoder      = encoder
        self.bank         = bank
        self.threshold    = pass_threshold
        self.current_step = 0
        self.history      = []
        self._mode        = bank._mode   # 'xfeat' | 'lightglue' | 'dino'

    # ── Main entry ────────────────────────────────────────────────────────────

    def verify(self, frame) -> dict:
        """
        Verify a live frame against the expected SOP step.

        frame : PIL.Image or np.ndarray (OpenCV BGR)

        Returns:
          {
            'expected_step'    : int
            'expected_name'    : str
            'similarity'       : float   ← best score among all refs
            'passed'           : bool
            'best_ref_idx'     : int     ← which reference image matched best
            'message'          : str
            # XFeat only:
            'inliers'          : int
            'total_matches'    : int
            'mconf_mean'       : float
            'mkpts_ref'        : np.ndarray (M, 2)
            'mkpts_live'       : np.ndarray (M, 2)
            # DINOv2/CNN only:
            'all_similarities' : dict {step_name: score}
          }
        """
        if self._mode in ('xfeat', 'lightglue'):
            result = self._verify_xfeat(frame)
        else:
            result = self._verify_dino(frame)

        self.history.append(result)
        return result

    # ── XFeat / LightGlue verification ───────────────────────────────────────

    def _verify_xfeat(self, frame) -> dict:
        feat_live = self.encoder.encode(frame)
        expected  = self.current_step

        # ── Match against all refs of the expected step ────────────────────
        best_result  = None
        best_ref_idx = 0

        ref_feats_for_step = self.bank.get_refs_for_step(expected)
        # ref_feats_for_step = list of feature dicts (one per reference image)
        ic(len(self.bank.embeddings))
        for ref_idx, ref_feat in enumerate(self.bank.embeddings):
            m = self.encoder.match(ref_feat[ref_idx], feat_live)
            ic(ref_idx, m['similarity'])
            if best_result is None or m['similarity'] > best_result['similarity']:
                best_result  = m
                best_ref_idx = ref_idx

        if best_result is None:
            return self._empty_result()

        ic(best_result['inliers'],
           best_result['total_matches'],
           best_result['similarity'],
           best_ref_idx,
           self.threshold)

        passed = best_result['similarity'] >= self.threshold

        result = {
            'expected_step': expected,
            'expected_name': self.bank.steps[expected]['step_name'],
            'similarity':    float(best_result['similarity']),
            'mconf_mean':    float(best_result.get('mconf_mean', 0.0)),
            'passed':        passed,
            'best_ref_idx':  best_ref_idx,
            'inliers':       best_result['inliers'],
            'total_matches': best_result['total_matches'],
            'mkpts_ref':     best_result['mkpts_ref'],
            'mkpts_live':    best_result['mkpts_live'],
        }

        num_refs = self.bank.num_refs(expected)
        if passed:
            result['message'] = (
                f"PASS — {result['expected_name']} "
                f"(ref {best_ref_idx+1}/{num_refs}, "
                f"{best_result['inliers']} inliers, "
                f"sim={best_result['similarity']:.3f})"
            )
        else:
            result['message'] = (
                f"NOT RECOGNISED — best ref {best_ref_idx+1}/{num_refs}, "
                f"{best_result['inliers']} inliers, "
                f"sim={best_result['similarity']:.3f} "
                f"(need ≥ {self.threshold:.2f})"
            )

        return result

    # ── DINOv2 / CNN verification ─────────────────────────────────────────────

    def _verify_dino(self, frame) -> dict:
        z_live   = self.encoder.encode(frame)          # (dim,)
        expected = self.current_step

        # ── For each step, score against all refs → take max ──────────────
        step_scores = []
        step_best_ref_idxs = []

        for step_idx in range(len(self.bank.steps)):
            refs = self.bank.get_refs_for_step(step_idx)  # (N_refs, dim)
            # cosine sim: dot product (already L2 normalized)
            sims = refs @ z_live                           # (N_refs,)
            best_ref = int(np.argmax(sims))
            step_scores.append(float(sims[best_ref]))
            step_best_ref_idxs.append(best_ref)

        best_step    = int(np.argmax(step_scores))
        best_sim     = step_scores[best_step]
        best_ref_idx = step_best_ref_idxs[best_step]

        # Expected step score
        expected_sim = step_scores[expected]
        passed       = expected_sim >= self.threshold

        ic(expected_sim, best_sim, best_step, best_ref_idx, self.threshold)

        result = {
            'expected_step':    expected,
            'expected_name':    self.bank.steps[expected]['step_name'],
            'similarity':       round(expected_sim, 4),
            'passed':           passed,
            'best_ref_idx':     step_best_ref_idxs[expected],   # best ref within expected step
            'best_match_step':  best_step,
            'best_match_name':  self.bank.steps[best_step]['step_name'],
            'best_match_sim':   round(best_sim, 4),
            'all_similarities': {
                self.bank.steps[i]['step_name']: round(s, 4)
                for i, s in enumerate(step_scores)
            },
        }

        if passed:
            num_refs = self.bank.num_refs(expected)
            result['message'] = (
                f"PASS — {result['expected_name']} "
                f"(ref {result['best_ref_idx']+1}/{num_refs}, "
                f"sim={expected_sim:.3f})"
            )
        else:
            if best_step != expected and best_sim >= self.threshold:
                result['message'] = (
                    f"WRONG STEP — detected '{result['best_match_name']}' "
                    f"(sim={best_sim:.3f}) but expected '{result['expected_name']}'"
                )
            else:
                result['message'] = (
                    f"NOT RECOGNISED — sim={expected_sim:.3f} "
                    f"(need ≥ {self.threshold:.2f})"
                )

        return result

    # ── Utility ───────────────────────────────────────────────────────────────

    def reset(self):
        self.current_step = 0
        self.history      = []

    def jump_to_step(self, step_id: int):
        self.current_step = step_id

    def advance_step(self):
        """Move to the next step after a PASS."""
        if self.current_step < len(self.bank.steps) - 1:
            self.current_step += 1
            return True
        return False   # already at last step

    def _empty_result(self) -> dict:
        expected = self.current_step
        return {
            'expected_step':   expected,
            'expected_name':   self.bank.steps[expected]['step_name'] if self.bank.steps else '',
            'similarity':      0.0,
            'mconf_mean':      0.0,
            'passed':          False,
            'best_ref_idx':    0,
            'inliers':         0,
            'total_matches':   0,
            'mkpts_ref':       np.empty((0, 2)),
            'mkpts_live':      np.empty((0, 2)),
            'message':         'NOT RECOGNISED — no features extracted',
        }