
from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp

from config import AppConfig, SOPStep
from sop_engine import SOPEngine, FlashMessage
from hand_tracker import HandState


class Renderer:

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_hands = mp.solutions.hands
        self._conn_style = self._mp_drawing.DrawingSpec(
            color=cfg.colors.accent, thickness=2)
        self._lm_style   = self._mp_drawing.DrawingSpec(
            color=cfg.colors.green, thickness=4, circle_radius=4)

    def draw_frame(self, display: np.ndarray, engine: SOPEngine,
                   hand: HandState, flash: FlashMessage | None, fps: float):
        h, w = display.shape[:2]

        if not engine.all_done:
            self._draw_all_zones(display, engine)
            self._draw_hands(display, hand)
            self._draw_grip_label(display, hand)
            self._draw_step_hint(display, engine)
            self._draw_inspect_crop(display, engine)   # new: highlight crop region

        self._draw_step_progress(display, engine, w)

        if flash:
            self._flash_result(display, flash)

        if engine.all_done:
            self._draw_all_done(display, w, h, engine)

        self._draw_instruction_bar(display, engine, h, w)
        self._draw_fps(display, fps)

    # ── Zone drawing ───────────────────────────────────────────────────────────

    def _draw_hands(self, display: np.ndarray, hand: HandState):
        if hasattr(hand, 'raw_landmarks') and hand.raw_landmarks:
            for lms in hand.raw_landmarks:
                self._mp_drawing.draw_landmarks(
                    display, lms, self._mp_hands.HAND_CONNECTIONS,
                    self._lm_style, self._conn_style,
                )

    def _draw_all_zones(self, display: np.ndarray, engine: SOPEngine):
        cfg = self._cfg
        rt  = engine.runtime

        for i, s in enumerate(cfg.sop_steps):
            color  = s.clr_pick if i == engine.current_step else cfg.colors.gray
            label  = f"PICK S{i+1}" if i >= engine.current_step else f"S{i+1} DONE"
            active = (i == engine.current_step and not rt.picked)
            self._draw_zone(display, s.zone_pick, color, label, active)

        self._draw_zone(display, cfg.assembly_zone, cfg.colors.green, "ASSEMBLY",
                        active=True)

    def _draw_zone(self, frame: np.ndarray, zone: tuple, color: tuple,
                   label: str, active: bool = False):
        x1, y1, x2, y2 = zone
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if active else 1)
        if active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, label, (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ── Inspect crop region highlight ──────────────────────────────────────────

    def _draw_inspect_crop(self, display: np.ndarray, engine: SOPEngine):
        """
        For inspect-mode steps: draw a dashed rectangle around the crop region
        and show the live similarity score (or status) above it.
        crop_coords in config are (y1, y2, x1, x2) — convert for cv2.
        """
        step = engine.step_cfg
        rt   = engine.runtime

        if not step.needs_inspect:
            return

        ins = step.inspect
        cy1, cy2, cx1, cx2 = ins.crop_coords   # config format: y1,y2,x1,x2
        cfg = self._cfg

        if rt.inspect_passed:
            color = cfg.colors.green
            label = "INSPECT: PASS"
        elif rt.inspect_result is not None:
            sim = rt.inspect_result.get("similarity", 0.0)
            thr = ins.pass_threshold if ins.pass_threshold is not None \
                  else cfg.dino.pass_threshold
            color = cfg.colors.red
            label = f"INSPECT: {sim:.2f} / {thr:.2f}"
        else:
            color = cfg.colors.yellow
            label = "INSPECT: waiting..."

        # Draw dashed rect using cv2 coords (x1,y1) → (x2,y2)
        self._draw_dashed_rect(display, cx1, cy1, cx2, cy2, color)
        cv2.putText(display, label, (cx1 + 4, cy1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def _draw_dashed_rect(self, frame, x1, y1, x2, y2, color,
                          dash=10, thickness=2):
        """Draw a dashed rectangle."""
        pts = [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]
        for (sx, sy), (ex, ey) in pts:
            length = int(((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5)
            steps  = max(1, length // (dash * 2))
            for k in range(steps):
                t0 = (2 * k * dash) / length
                t1 = min((2 * k + 1) * dash / length, 1.0)
                p0 = (int(sx + t0 * (ex - sx)), int(sy + t0 * (ey - sy)))
                p1 = (int(sx + t1 * (ex - sx)), int(sy + t1 * (ey - sy)))
                cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)

    # ── Labels ─────────────────────────────────────────────────────────────────

    def _draw_label(self, frame: np.ndarray, text: str, color: tuple,
                    x: int = 14, y: int = 100):
        tw = len(text) * 9 + 8
        cv2.rectangle(frame, (x - 2, y - 18), (x + tw, y + 6), (15, 15, 25), -1)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def _draw_grip_label(self, display: np.ndarray, hand: HandState):
        label = "GRIP" if hand.grip else "OPEN"
        color = self._cfg.colors.green if hand.grip else self._cfg.colors.yellow
        self._draw_label(display, label, color, x=14, y=140)

    def _draw_step_hint(self, display: np.ndarray, engine: SOPEngine):
        rt   = engine.runtime
        step = engine.step_cfg

        if not rt.picked:
            self._draw_label(
                display,
                f"[{step.name}] Go to PICK zone + grip",
                self._cfg.colors.accent, x=14, y=100,
            )
        elif step.needs_inspect and not rt.inspect_passed:
            self._draw_label(
                display,
                f"[{step.name}] Hold item in view for inspection",
                self._cfg.colors.yellow, x=14, y=100,
            )

    # ── Progress panel ─────────────────────────────────────────────────────────

    def _draw_step_progress(self, frame: np.ndarray, engine: SOPEngine, w: int):
        cfg     = self._cfg
        panel_x = w - 230
        for i, step in enumerate(cfg.sop_steps):
            y = 20 + i * 28
            if   i < engine.current_step:  color, symbol = cfg.colors.green,  "DONE"
            elif i == engine.current_step: color, symbol = cfg.colors.yellow, "NOW >"
            else:                          color, symbol = cfg.colors.gray,   "LOCK"

            # Mode badge
            mode_tag = " [V]" if step.needs_inspect else ""
            cv2.putText(frame, f"{step.name}{mode_tag} [{symbol}]", (panel_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # ── Flash banner ───────────────────────────────────────────────────────────

    def _flash_result(self, frame: np.ndarray, flash: FlashMessage):
        if flash.passed is True:
            color = self._cfg.colors.green
        elif flash.passed is False:
            color = self._cfg.colors.red
        else:
            color = self._cfg.colors.accent   # neutral / progress

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), color, -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, flash.text[:65], (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self._cfg.colors.white, 2, cv2.LINE_AA)

    # ── All done overlay ───────────────────────────────────────────────────────

    def _draw_all_done(self, display: np.ndarray, w: int, h: int, engine: SOPEngine):
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 180, 60), -1)
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
        cv2.putText(display, "ALL STEPS COMPLETE!", (60, h // 2 - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, self._cfg.colors.white, 2, cv2.LINE_AA)
        elapsed = engine._elapsed_total
        if elapsed > 0:
            cv2.putText(display, f"Total time: {elapsed:.1f}s",
                        (160, h // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self._cfg.colors.accent, 1, cv2.LINE_AA)
        cv2.putText(display, "Press R to reset", (160, h // 2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self._cfg.colors.yellow, 1, cv2.LINE_AA)

    # ── Bottom bar & FPS ───────────────────────────────────────────────────────

    def _draw_instruction_bar(self, display: np.ndarray, engine: SOPEngine,
                               h: int, w: int):
        text = ("ALL DONE — press R to reset" if engine.all_done
                else engine.step_cfg.instruction)
        cv2.rectangle(display, (0, h - 30), (w, h), (15, 15, 25), -1)
        cv2.putText(display, text, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self._cfg.colors.white, 1, cv2.LINE_AA)

    def _draw_fps(self, display: np.ndarray, fps: float):
        cv2.putText(display, f"FPS {fps:.1f}", (14, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, self._cfg.colors.green, 1, cv2.LINE_AA)