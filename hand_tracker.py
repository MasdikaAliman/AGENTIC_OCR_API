"""
hand_tracker.py — wraps MediaPipe Hands and exposes per-frame hand state.
No drawing, no SOP logic — purely detects gesture and zone membership.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mediapipe as mp
import cv2

from config import AppConfig
from icecream import ic
# ── Landmark index groups ──────────────────────────────────────────────────────
_FINGER_TIPS = [4, 8, 12, 16, 20]
_FINGER_MCP  = [6, 10, 14]
_PALM_ANCHORS = [0, 5, 9, 13, 17]

# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class HandState:
    grip:          bool = False
    in_pick:       bool = False
    in_assembly:   bool = False
    in_wrong_zone: bool = False
    is_two_hands_near: bool = False
    hands_distance: float = 0.0
    auto_hand_step: int = 0 # This for auto define step based on hand

    def merge(self, other: "HandState") -> None:
        """OR-merge another HandState into this one (any hand triggering counts)."""
        self.grip          |= other.grip
        self.in_pick       |= other.in_pick
        self.in_assembly   |= other.in_assembly
        self.in_wrong_zone |= other.in_wrong_zone
        self.is_two_hands_near |= other.is_two_hands_near
        self.hands_distance = other.hands_distance
        self.auto_hand_step |= other.auto_hand_step

# ── Geometry helpers (module-private) ─────────────────────────────────────────

def _landmark_pixels(lms, indices: list[int], w: int, h: int) -> list[tuple[int, int]]:
    return [(int(lms.landmark[i].x * w), int(lms.landmark[i].y * h)) for i in indices]

def _point_in_zone(px: int, py: int, zone: tuple) -> bool:
    x1, y1, x2, y2 = zone
    return x1 <= px <= x2 and y1 <= py <= y2
    # x1, y1, x2, y2 = zone
    # cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # # radius = fraction of the shorter side
    # r = min(x2 - x1, y2 - y1) * 0.8
    # return (px - cx) ** 2 + (py - cy) ** 2 <= r * r

def _any_point_in_zone(points: list[tuple], zone: tuple) -> bool:
    return any(_point_in_zone(px, py, zone) for px, py in points)


def _hand_centroid(points: list[tuple]) -> tuple[float, float]:
    """Average position of all given landmark points — stable, ignores finger strays."""
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _centroid_in_zone(points: list[tuple], zone: tuple, margin: int) -> bool:
    """
    Check if the hand centroid sits inside a zone shrunk inward by `margin` px.
    Centroid  → stable single point (no edge-grazing from individual fingers).
    Shrunk box → hand must be clearly inside, not just touching the boundary.
    """
    cx, cy = _hand_centroid(points)
    x1, y1, x2, y2 = zone
    return (x1 + margin) <= cx <= (x2 - margin) and \
        (y1 + margin) <= cy <= (y2 - margin)

def _centroids_distance(c1, c2):
    return float(((c1[0]- c2[0]) ** 2 + (c1[1]- c2[1]) ** 2) ** 0.5)


# ── HandTracker class ──────────────────────────────────────────────────────────

class HandTracker:
    """
    Wraps MediaPipe Hands.
    Call process(frame) each frame → returns HandState for the right hand.
    """

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        mp_hands  = mp.solutions.hands
        self._hands = mp_hands.Hands(
            model_complexity         = cfg.mediapipe.model_complexity,
            max_num_hands            = cfg.mediapipe.max_hands,
            min_detection_confidence = cfg.mediapipe.detection_confidence,
            min_tracking_confidence  = cfg.mediapipe.tracking_confidence,
        )
        self._mp_hands   = mp_hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._conn_style = self._mp_drawing.DrawingSpec(
            color=cfg.colors.accent, thickness=2)
        self._lm_style   = self._mp_drawing.DrawingSpec(
            color=cfg.colors.green, thickness=4, circle_radius=4)
        self.auto_current_step = 0
    
    def process(self, frame: np.ndarray, display: np.ndarray,
                current_step: int) -> HandState:
        """
        Run MediaPipe on , draw landmarks onto .

        Two-pass strategy:
          Pass 1 : draw landmarks + collect per-hand state and centroid.
          Pass 2 : if two hands detected, compute proximity between their
                   centroids and set is_two_hands_near / hands_distance on merged state.
        """
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        merged = HandState()

        if not (result.multi_hand_landmarks and result.multi_handedness):
            return merged

        h, w = display.shape[:2]

        # ── Pass 1: per-hand analysis ──────────────────────────────────────
        centroids: list[tuple[float, float]] = []

        for lms, handedness in zip(result.multi_hand_landmarks,
                                   result.multi_handedness):
            self._mp_drawing.draw_landmarks(
                display, lms, self._mp_hands.HAND_CONNECTIONS,
                self._lm_style, self._conn_style,
            )

            state, centroid = self._analyse(lms, w, h, current_step)
            merged.merge(state)
            centroids.append(centroid)
        # ic(len(centroids))
        # ── Pass 2: two-hand proximity ─────────────────────────────────────
        if len(centroids) >= 2:
            # Use the first two detected hands (MediaPipe reports at most max_hands)
            dist = _centroids_distance(centroids[0], centroids[1])
            # ic(dist)
            merged.hands_distance = dist
            merged.is_two_hands_near = dist <= 100
        else:
            merged.hands_distance = 0.0
            merged.is_two_hands_near     = False

        return merged

    # ── Private ────────────────────────────────────────────────────────────────

    def _is_grip(self, lms) -> bool:
        pts        = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark])
        hand_scale = np.linalg.norm(pts[0] - pts[9])
        if hand_scale == 0:
            return False
        center       = (pts[0] + pts[5] + pts[17]) / 3
        close_finger = sum(
            1 for tip in _FINGER_TIPS[1:]
            if np.linalg.norm(pts[tip] - center) < self._cfg.gesture.grip_threshold
        )
        return close_finger >= 3

    def _analyse(self, lms, w: int, h: int, current_step: int) -> HandState:
        """Single-pass analysis for one right-hand landmark set."""
        mcps      = _landmark_pixels(lms, _FINGER_MCP, w, h)
        gripping  = self._is_grip(lms)
        # anchors  = _landmark_pixels(lms, _PALM_ANCHORS, w, h)
        # centroid = _hand_centroid(anchors)
        
        wrist_lm = lms.landmark[0]
        centroid = (int(wrist_lm.x * w), int(wrist_lm.y * h))

        # for step_id , zone in self._cfg.pick_zones.items():
        #     if _any_point_in_zone(mcps, zone=zone):
        #         self.auto_current_step = step_id

        # ic(self.auto_current_step)

        pick_zone = self._cfg.pick_zones[current_step]
        # in_pick   = _any_point_in_zone(mcps, pick_zone)
        # in_asm    = _any_point_in_zone(mcps, self._cfg.assembly_zone)
        in_pick = _centroid_in_zone(mcps, pick_zone, 5)
        in_asm = _centroid_in_zone(mcps, self._cfg.assembly_zone, 5)

        # Wrong-zone: only evaluated when gripping outside the correct zone
        in_wrong = False
        if gripping and not in_pick:
            in_wrong = any(
                _any_point_in_zone(mcps, zone)
                for step_id, zone in self._cfg.pick_zones.items()
                if step_id != current_step
            )

        state = HandState(
            grip          = gripping,
            in_pick       = in_pick,
            in_assembly   = in_asm,
            in_wrong_zone = in_wrong,
            auto_hand_step= self.auto_current_step
        )

        return state, centroid

    def close(self):
        self._hands.close()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
