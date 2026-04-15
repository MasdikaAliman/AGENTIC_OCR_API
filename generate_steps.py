"""ONTROLS:
  Left-click + drag  : draw a rectangle (zone_pick OR crop_coords)
  M                  : toggle step mode  hand_only ↔ inspect
  R                  : redraw / reset current rectangle
  N                  : confirm current step → move to next step
  D                  : delete last confirmed step
  C                  : (inspect mode) capture reference image for this step
  A                  : draw assembly zone  (only needed once, shared by all steps)
  SPACE              : finish — show YAML preview
  Q / ESC            : quit without saving

WORKFLOW PER STEP:
  hand_only:
    1. Draw zone_pick rectangle
    2. Press N to confirm

  inspect:
    1. Press M to switch to inspect mode
    2. Draw zone_pick rectangle
    3. Press M again to switch to crop drawing
    4. Draw crop_coords rectangle
    5. Press C to capture reference image
    6. Press N to confirm

OUTPUT:
  Writes the sop_steps section (and assembly_zone) into config.yaml.
  Existing camera/mediapipe/gesture/colors/dino sections are preserved.
"""

from __future__ import annotations
import copy
import os
import time
import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ── Config ─────────────────────────────────────────────────────────────────────

CONFIG_PATH     = "SOP_CUSTOM.yaml"
DISPLAY_W       = 1280
DISPLAY_H       = 720
REF_IMAGE_BASE  = "data/sop_ref"   # reference images saved here per step

# Step color palette (cycles through for each step)
STEP_COLORS = [
    (60,  60,  220),   # red
    (0,   140, 255),   # orange
    (200, 80,  200),   # purple
    (50,  220, 90),    # green
    (0,   210, 255),   # accent
    (30,  220, 200),   # yellow
]
COLOR_NAMES = ["red", "orange", "purple", "green", "accent", "yellow"]

ASSEMBLY_COLOR = (50, 220, 90)   # green

# ── Drawing states ─────────────────────────────────────────────────────────────

@dataclass
class DrawState:
    """Tracks the current mouse-drawing state."""
    drawing:    bool             = False
    start:      tuple            = (0, 0)
    end:        tuple            = (0, 0)
    rect:       Optional[tuple]  = None   # finalised (x1,y1,x2,y2)

    def reset(self):
        self.drawing = False
        self.start   = (0, 0)
        self.end     = (0, 0)
        self.rect    = None


# ── Step draft ─────────────────────────────────────────────────────────────────

@dataclass
class StepDraft:
    step_id:          int
    name:             str             = ""
    mode:             str             = "hand_only"   # "hand_only" | "inspect"
    zone_pick:        Optional[tuple] = None          # (x1,y1,x2,y2)
    crop_coords:      Optional[tuple] = None          # (y1,y2,x1,x2)
    reference_folder: Optional[str]   = None
    ref_images:       list            = field(default_factory=list)  # captured paths

    @property
    def is_complete(self) -> bool:
        if self.zone_pick is None:
            return False
        if self.mode == "inspect":
            return (self.crop_coords is not None and
                    len(self.ref_images) > 0)
        return True


# ── Global state ───────────────────────────────────────────────────────────────

class Generator:

    def __init__(self):
        self.steps:         list[StepDraft] = []
        self.assembly_zone: Optional[tuple] = None

        # Current step being built
        self.current        = StepDraft(step_id=0)
        self.draw           = DrawState()

        # What the mouse is currently targeting
        # "zone_pick" | "crop_coords" | "assembly"
        self.draw_target    = "zone_pick"

        self.cap            = None
        self.last_frame     = None

        # Pause state — when True, camera is frozen and drawing still works
        self.paused         = False
        self.frozen_frame   = None   # copy of frame at the moment P was pressed

    # ── Camera ────────────────────────────────────────────────────────────────

    def open_camera(self, source=0, frame_width=3840, frame_height=2160):
        print(f"[INFO] Opening camera source {source}...")
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    def read_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            return self.last_frame
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        self.last_frame = frame.copy()
        return frame

    # ── Mouse callback ────────────────────────────────────────────────────────

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw.drawing = True
            self.draw.start   = (x, y)
            self.draw.end     = (x, y)
            self.draw.rect    = None

        elif event == cv2.EVENT_MOUSEMOVE and self.draw.drawing:
            self.draw.end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.draw.drawing = False
            self.draw.end     = (x, y)
            x1 = min(self.draw.start[0], self.draw.end[0])
            y1 = min(self.draw.start[1], self.draw.end[1])
            x2 = max(self.draw.start[0], self.draw.end[0])
            y2 = max(self.draw.start[1], self.draw.end[1])
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                self.draw.rect = (x1, y1, x2, y2)
                self._apply_rect()

    def _apply_rect(self):
        r = self.draw.rect
        if self.draw_target == "zone_pick":
            self.current.zone_pick = r
        elif self.draw_target == "crop_coords":
            # crop_coords stored as (y1,y2,x1,x2) to match existing code convention
            x1, y1, x2, y2 = r
            self.current.crop_coords = (y1, y2, x1, x2)
        elif self.draw_target == "assembly":
            self.assembly_zone = r

    # ── Key handling ──────────────────────────────────────────────────────────

    def handle_key(self, key: int) -> str:
        """Returns 'continue', 'finish', or 'quit'."""

        # M — toggle mode or switch draw target
        if key == ord('m'):
            if self.current.mode == "hand_only":
                self.current.mode = "inspect"
                self.draw_target  = "zone_pick"
                print(f"[MODE] Step {self.current.step_id+1} → INSPECT")
            elif self.draw_target == "zone_pick":
                self.draw_target = "crop_coords"
                print(f"[DRAW] Now drawing: crop_coords")
            else:
                self.draw_target = "zone_pick"
                print(f"[DRAW] Now drawing: zone_pick")

        elif key == ord("b"):
            if self.current.mode == "inspect":
                self.current.mode = "hand_only"
                self.draw_target = "zone_pick"
                print(f"[MODE] Step {self.current.step_id+1} → HAND_ONLY")

        # A — switch to drawing assembly zone
        elif key == ord('a'):
            self.draw_target = "assembly"
            self.draw.reset()
            print("[DRAW] Now drawing: assembly zone (shared by all steps)")

        # R — reset current rectangle
        elif key == ord('r'):
            if self.draw_target == "zone_pick":
                self.current.zone_pick   = None
            elif self.draw_target == "crop_coords":
                self.current.crop_coords = None
            elif self.draw_target == "assembly":
                self.assembly_zone       = None
            self.draw.reset()
            print(f"[RESET] Cleared {self.draw_target}")

        # C — capture reference image (inspect mode)
        elif key == ord('c'):
            if self.current.mode != "inspect":
                print("[WARN] C only works in inspect mode (press M first)")
            elif self.last_frame is None:
                print("[WARN] No frame available")
            else:
                self._capture_reference_image()

        # N — confirm current step
        elif key == ord('n'):
            return self._confirm_step()

        # D — delete last confirmed step
        elif key == ord('d'):
            if self.steps:
                removed = self.steps.pop()
                print(f"[DELETE] Removed step: {removed.name}")
                # restore as current
                self.current      = removed
                self.draw_target  = "zone_pick"
            else:
                print("[WARN] No steps to delete")


        # P — toggle pause (freeze frame, drawing still works)
        elif key == ord('p'):
            self.paused = not self.paused
            if self.paused:
                self.frozen_frame = self.last_frame.copy() if self.last_frame is not None else None
                print("[PAUSE] Frozen — draw zones on this frame. Press P to resume.")
            else:
                self.frozen_frame = None
                print("[RESUME] Camera live again.")

        # SPACE — finish
        elif key == ord(' '):
            return 'finish'

        # Q / ESC — quit
        elif key in (ord('q'), 27):
            return 'quit'

        return 'continue'

    def _capture_reference_image(self):
        step_id    = self.current.step_id
        folder     = Path(REF_IMAGE_BASE) / f"step{step_id}" / "correct"
        folder.mkdir(parents=True, exist_ok=True)

        idx      = len(self.current.ref_images) + 1
        filename = folder / f"ref_{idx:02d}.jpg"

        # If crop is defined, save the cropped region; else full frame
        frame = self.last_frame.copy()
        if self.current.crop_coords is not None:
            y1, y2, x1, x2 = self.current.crop_coords
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(str(filename), crop)
                print(f"[CAPTURE] Saved crop to '{filename}'")
            else:
                print("[WARN] Crop region is empty — check crop_coords rectangle")
                return
        else:
            cv2.imwrite(str(filename), frame)
            print(f"[CAPTURE] Saved full frame to '{filename}'")

        self.current.ref_images.append(str(filename))
        self.current.reference_folder = str(folder)

    def _confirm_step(self) -> str:
        c = self.current

        if c.zone_pick is None:
            print("[WARN] Draw zone_pick rectangle first, then press N")
            return 'continue'

        if c.mode == "inspect":
            if c.crop_coords is None:
                print("[WARN] Inspect mode: press M to switch to crop_coords drawing, draw it, then press N")
                return 'continue'
            if len(c.ref_images) == 0:
                print("[WARN] Inspect mode: press C to capture at least one reference image first")
                return 'continue'

        # Auto-name
        c.name = f"STEP {c.step_id + 1}"
        self.steps.append(c)
        print(f"[CONFIRM] {c.name} ({c.mode}) — zone_pick={c.zone_pick}")

        # Start fresh step
        next_id       = c.step_id + 1
        self.current  = StepDraft(step_id=next_id)
        self.draw.reset()
        self.draw_target = "zone_pick"
        print(f"[NEXT] Now configuring STEP {next_id + 1}")
        return 'continue'

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        h, w    = display.shape[:2]

        # Draw all confirmed steps
        for s in self.steps:
            color = self._step_color(s.step_id)
            self._draw_zone(display, s.zone_pick, color,
                            f"S{s.step_id+1} PICK ({s.mode[0].upper()})", active=False)
            if s.crop_coords is not None:
                y1, y2, x1, x2 = s.crop_coords
                self._draw_dashed_rect(display, x1, y1, x2, y2, color)

        # Draw assembly zone
        if self.assembly_zone is not None:
            self._draw_zone(display, self.assembly_zone, ASSEMBLY_COLOR,
                            "ASSEMBLY", active=True)

        # Draw current step in progress
        c     = self.current
        color = self._step_color(c.step_id)

        if c.zone_pick is not None:
            self._draw_zone(display, c.zone_pick, color,
                            f"S{c.step_id+1} PICK", active=(self.draw_target == "zone_pick"))

        if c.crop_coords is not None:
            y1, y2, x1, x2 = c.crop_coords
            self._draw_dashed_rect(display, x1, y1, x2, y2, (0, 210, 255))
            cv2.putText(display, f"CROP", (x1+4, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 210, 255), 1, cv2.LINE_AA)

        # Live rectangle being drawn
        if self.draw.drawing or (self.draw.rect is not None and
                                  self.draw.start != (0, 0)):
            ex, ey = self.draw.end if self.draw.drawing else \
                     (self.draw.rect[2], self.draw.rect[3]) if self.draw.rect else (0, 0)
            sx, sy = self.draw.start
            if ex != 0 or ey != 0:
                live_color = (0, 210, 255) if self.draw_target == "crop_coords" \
                             else ASSEMBLY_COLOR if self.draw_target == "assembly" \
                             else color
                cv2.rectangle(display, (min(sx,ex), min(sy,ey)),
                              (max(sx,ex), max(sy,ey)), live_color, 2)

        # Pause overlay
        if self.paused:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            cv2.putText(display, "PAUSED", (w//2 - 60, h//2 - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 210, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "Draw zones freely  |  P = resume",
                        (w//2 - 140, h//2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # HUD
        self._draw_hud(display, h, w)
        return display

    def _draw_hud(self, display: np.ndarray, h: int, w: int):
        c    = self.current
        mode = c.mode.upper()
        tgt  = self.draw_target.upper().replace("_", " ")

        # Status bar
        cv2.rectangle(display, (0, 0), (w, 28), (20, 20, 30), -1)
        cv2.putText(display,
                    f"STEP {c.step_id+1} | MODE: {mode} | DRAWING: {tgt} | "
                    f"Confirmed: {len(self.steps)}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (240, 240, 240), 1, cv2.LINE_AA)

        # Checklist for current step
        checks = []
        checks.append(("zone_pick",  c.zone_pick   is not None))
        if c.mode == "inspect":
            checks.append(("crop_coords",  c.crop_coords is not None))
            checks.append((f"ref imgs ({len(c.ref_images)})", len(c.ref_images) > 0))
        if self.assembly_zone is None:
            checks.append(("assembly [press A]", False))

        y = 50
        for label, done in checks:
            mark  = "✓" if done else "○"
            color = (50, 220, 90) if done else (60, 60, 220)
            cv2.putText(display, f"{mark} {label}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            y += 18

        # Key hints at bottom
        hints = [
            "P: pause/resume",
            "M: toggle mode/target",
            "A: assembly zone",
            "R: reset rect",
            "C: capture ref",
            "N: next step",
            "D: delete last",
            "SPACE: finish",
        ]
        cv2.rectangle(display, (0, h-22), (w, h), (20, 20, 30), -1)
        cv2.putText(display, "  |  ".join(hints), (4, h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_zone(self, frame, zone, color, label, active=False):
        x1, y1, x2, y2 = zone
        thickness = 2 if active else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.putText(frame, label, (x1+4, y1+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def _draw_dashed_rect(self, frame, x1, y1, x2, y2, color, dash=8, thickness=1):
        pts = [((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
               ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))]
        for (sx,sy),(ex,ey) in pts:
            length = max(int(((ex-sx)**2+(ey-sy)**2)**0.5), 1)
            steps  = max(1, length // (dash*2))
            for k in range(steps):
                t0 = (2*k*dash)/length
                t1 = min((2*k+1)*dash/length, 1.0)
                p0 = (int(sx+t0*(ex-sx)), int(sy+t0*(ey-sy)))
                p1 = (int(sx+t1*(ex-sx)), int(sy+t1*(ey-sy)))
                cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _step_color(step_id: int) -> tuple:
        return STEP_COLORS[step_id % len(STEP_COLORS)]

    # ── YAML generation ───────────────────────────────────────────────────────

    def build_yaml_preview(self) -> str:
        """Build the full updated config.yaml as a string for preview."""
        raw = _load_existing_config()

        # Update assembly zone
        if self.assembly_zone is not None:
            raw.setdefault("zones", {})["assembly"] = list(self.assembly_zone)

        # Build sop_steps
        sop_steps = []
        for s in self.steps:
            color_name = COLOR_NAMES[s.step_id % len(COLOR_NAMES)]
            entry = {
                "step_id":     s.step_id,
                "name":        s.name,
                "instruction": f"Ambil item dari area {color_name.upper()}",
                "zone_pick":   list(s.zone_pick),
                "clr_pick":    color_name,
                "mode":        s.mode,
            }
            if s.mode == "inspect" and s.crop_coords is not None:
                entry["inspect"] = {
                    "crop_coords":      list(s.crop_coords),
                    "reference_folder": s.reference_folder or
                                        f"{REF_IMAGE_BASE}/step{s.step_id}/correct",
                }
            sop_steps.append(entry)

        raw["sop_steps"] = sop_steps

        return yaml.dump(raw, default_flow_style=False,
                         allow_unicode=True, sort_keys=False)

    def save_config(self):
        """Write updated config.yaml to disk."""
        raw = _load_existing_config()

        if self.assembly_zone is not None:
            raw.setdefault("zones", {})["assembly"] = list(self.assembly_zone)

        sop_steps = []
        for s in self.steps:
            color_name = COLOR_NAMES[s.step_id % len(COLOR_NAMES)]
            entry = {
                "step_id":     s.step_id,
                "name":        s.name,
                "instruction": f"Ambil item dari area {color_name.upper()}",
                "zone_pick":   list(s.zone_pick),
                "clr_pick":    color_name,
                "mode":        s.mode,
            }
            if s.mode == "inspect" and s.crop_coords is not None:
                entry["inspect"] = {
                    "crop_coords":      list(s.crop_coords),
                    "reference_folder": s.reference_folder or
                                        f"{REF_IMAGE_BASE}/step{s.step_id}/correct",
                }
            sop_steps.append(entry)

        raw["sop_steps"] = sop_steps

        with open(CONFIG_PATH, "w") as f:
            yaml.dump(raw, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
        print(f"\n[SAVED] Written to '{CONFIG_PATH}'")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_existing_config() -> dict:
    """Load existing config.yaml, return empty dict if not found."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def _preview_and_confirm(gen: Generator) -> bool:
    """Print YAML preview, ask user to confirm save. Returns True if saved."""
    yaml_str = gen.build_yaml_preview()

    print("\n" + "═" * 60)
    print("  YAML PREVIEW — sop_steps that will be written:")
    print("═" * 60)
    print(yaml_str)
    print("═" * 60)
    print(f"\n  {len(gen.steps)} step(s) configured:")
    for s in gen.steps:
        print(f"    STEP {s.step_id+1}  mode={s.mode}  "
              f"zone_pick={s.zone_pick}  "
              + (f"crop={s.crop_coords}  refs={len(s.ref_images)}"
                 if s.mode == "inspect" else ""))

    if gen.assembly_zone:
        print(f"  Assembly zone : {gen.assembly_zone}")
    else:
        print("  Assembly zone : NOT SET (will keep existing value in config.yaml)")

    print("\nSave to config.yaml? [Y = yes / any other key = discard]  ", end="", flush=True)
    ans = input().strip().lower()
    if ans == "y":
        gen.save_config()
        return True
    else:
        print("[DISCARDED] No changes written.")
        return False


# ── Main loop ──────────────────────────────────────────────────────────────────

def run():
    # Load camera source from config.yaml if it exists
    raw    = _load_existing_config()
    source = raw.get("camera", {}).get("source", 0)
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass\

    print(f"[INFO] Using camera source: {source} (from config.yaml)")
    gen = Generator()
    gen.open_camera(source)

    cv2.namedWindow("SOP Generator")
    cv2.setMouseCallback("SOP Generator", gen.mouse_callback)

    print("\n" + "═"*60)
    print("  SOP CONFIG GENERATOR")
    print("═"*60)
    print("  P          : pause camera (freeze frame) — draw zones on still image")
    print("  M          : toggle mode (hand_only ↔ inspect) / switch draw target")
    print("  A          : draw assembly zone (shared, do this once)")
    print("  R          : reset current rectangle")
    print("  C          : capture reference image  (inspect mode only)")
    print("  N          : confirm step → next step")
    print("  D          : delete last confirmed step")
    print("  SPACE      : finish and preview YAML")
    print("  Q / ESC    : quit without saving")
    print("═"*60)
    print(f"  Configuring STEP 1 | draw zone_pick, then press N\n")

    while True:
        # When paused use the frozen frame; otherwise read live from camera
        if gen.paused and gen.frozen_frame is not None:
            frame = gen.frozen_frame.copy()
        else:
            frame = gen.read_frame()
            if frame is None:
                print("[ERROR] Cannot read camera.")
                break

        display = gen.render(frame)
        cv2.imshow("SOP Generator", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue

        result = gen.handle_key(key)

        if result == "finish":
            if not gen.steps:
                print("[WARN] No steps confirmed yet. Confirm at least one step first.")
                continue
            cv2.destroyAllWindows()
            gen.cap.release()
            _preview_and_confirm(gen)
            return

        elif result == "quit":
            print("[QUIT] Exiting without saving.")
            break

    gen.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()