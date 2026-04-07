"""
config.py — loads config.yaml and exposes typed, validated settings.
All other modules import from here; nothing reads the YAML directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    source: str | int
    frame_w: int
    frame_h: int
    fps: int


@dataclass
class MediaPipeConfig:
    max_hands: int
    detection_confidence: float
    tracking_confidence: float
    model_complexity: int


@dataclass
class GestureConfig:
    grip_threshold: float
    success_delay: float
    pick_dwell_time: float


@dataclass
class ColorsConfig:
    white:  tuple
    green:  tuple
    red:    tuple
    yellow: tuple
    accent: tuple
    orange: tuple
    gray:   tuple
    purple: tuple

    def by_name(self, name: str) -> tuple:
        """Return a color tuple by string name (matches YAML clr_pick values)."""
        try:
            return getattr(self, name)
        except AttributeError:
            raise ValueError(f"Unknown color name '{name}'. "
                             f"Valid names: {list(self.__dataclass_fields__)}")


@dataclass
class SOPStep:
    step_id:     int
    name:        str
    instruction: str
    zone_pick:   tuple        # (x1, y1, x2, y2)
    clr_pick:    tuple        # BGR color tuple


@dataclass
class AppConfig:
    camera:    CameraConfig
    mediapipe: MediaPipeConfig
    gesture:   GestureConfig
    colors:    ColorsConfig
    assembly_zone: tuple      # (x1, y1, x2, y2)
    sop_steps: list[SOPStep]

    # Derived — built once for O(1) wrong-zone lookups
    pick_zones: dict[int, tuple] = field(init=False)

    def __post_init__(self):
        self.pick_zones = {s.step_id: s.zone_pick for s in self.sop_steps}


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """
    Parse config.yaml and return a fully validated AppConfig.
    Raises ValueError with a clear message if anything is wrong.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Colors must be parsed first — steps reference them by name
    raw_colors = raw["colors"]
    colors = ColorsConfig(**{k: tuple(v) for k, v in raw_colors.items()})

    # Camera source: keep as int if it looks like one, else string
    raw_cam = raw["camera"]
    source  = raw_cam["source"]
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    camera = CameraConfig(
        source  = source,
        frame_w = int(raw_cam["frame_w"]),
        frame_h = int(raw_cam["frame_h"]),
        fps     = int(raw_cam["fps"]),
    )

    mp_raw    = raw["mediapipe"]
    mediapipe = MediaPipeConfig(
        max_hands            = int(mp_raw["max_hands"]),
        detection_confidence = float(mp_raw["detection_confidence"]),
        tracking_confidence  = float(mp_raw["tracking_confidence"]),
        model_complexity     = int(mp_raw["model_complexity"]),
    )

    g_raw   = raw["gesture"]
    gesture = GestureConfig(
        grip_threshold = float(g_raw["grip_threshold"]),
        success_delay  = float(g_raw["success_delay"]),
        pick_dwell_time= float(g_raw["pick_dwell_time"])
    )

    assembly_zone = tuple(raw["zones"]["assembly"])

    # Parse SOP steps — validate step_id uniqueness
    seen_ids: set[int] = set()
    sop_steps: list[SOPStep] = []
    for i, s in enumerate(raw["sop_steps"]):
        sid = int(s["step_id"])
        if sid in seen_ids:
            raise ValueError(f"Duplicate step_id {sid} in sop_steps[{i}]")
        seen_ids.add(sid)

        sop_steps.append(SOPStep(
            step_id     = sid,
            name        = str(s["name"]),
            instruction = str(s["instruction"]),
            zone_pick   = tuple(s["zone_pick"]),
            clr_pick    = colors.by_name(s["clr_pick"]),
        ))

    # Sort by step_id to guarantee execution order
    sop_steps.sort(key=lambda s: s.step_id)

    return AppConfig(
        camera        = camera,
        mediapipe     = mediapipe,
        gesture       = gesture,
        colors        = colors,
        assembly_zone = assembly_zone,
        sop_steps     = sop_steps,
    )
