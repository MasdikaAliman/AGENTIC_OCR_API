"""
renderer.py — all OpenCV drawing in one place.
Receives state from SOPEngine and HandState; never mutates them.
"""

from __future__ import annotations
import cv2
import numpy as np

from config import AppConfig, SOPStep
from sop_engine import SOPEngine, FlashMessage
from hand_tracker import HandState


class Renderer:
    """Draws all overlays onto a display frame each tick."""

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg

    def draw_frame(self, display: np.ndarray, engine: SOPEngine,
                   hand: HandState, flash: FlashMessage | None, fps: float):
        """Master draw call — invoke once per frame after engine.update()."""
        h, w = display.shape[:2]

        if not engine.all_done:
            self._draw_all_zones(display, engine)
            self._draw_grip_label(display, hand)
            self._draw_step_hint(display, engine)

        self._draw_step_progress(display, engine, w)

        if flash:
            self._flash_result(display, flash)

        if engine.all_done:
            self._draw_all_done(display, w, h)

        self._draw_instruction_bar(display, engine, h, w)
        self._draw_fps(display, fps)

    # ── Zone drawing ───────────────────────────────────────────────────────────

    def _draw_all_zones(self, display: np.ndarray, engine: SOPEngine):
        cfg = self._cfg
        rt  = engine.runtime

        for i, s in enumerate(cfg.sop_steps):
            color  = s.clr_pick if i == engine.current_step else cfg.colors.gray
            label  = f"PICK S{i+1}" if i >= engine.current_step else f"S{i+1} DONE"
            active = (i == engine.current_step and not rt.picked)
            self._draw_zone(display, s.zone_pick, color, label, active)

        self._draw_zone(display, cfg.assembly_zone, cfg.colors.green, "ASSEMBLY", active=True)

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
        if not engine.runtime.picked:
            self._draw_label(
                display,
                f"[{engine.step_cfg.name}] Go to PICK zone + grip",
                self._cfg.colors.accent, x=14, y=100,
            )

    # ── Progress panel ─────────────────────────────────────────────────────────

    def _draw_step_progress(self, frame: np.ndarray, engine: SOPEngine, w: int):
        cfg     = self._cfg
        panel_x = w - 220
        for i, step in enumerate(cfg.sop_steps):
            y = 20 + i * 28
            if   i < engine.current_step:  color, symbol = cfg.colors.green,  "DONE"
            elif i == engine.current_step: color, symbol = cfg.colors.yellow, "NOW >"
            else:                          color, symbol = cfg.colors.gray,   "LOCK"
            cv2.putText(frame, f"{step.name} [{symbol}]", (panel_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # ── Flash banner ───────────────────────────────────────────────────────────

    def _flash_result(self, frame: np.ndarray, flash: FlashMessage):
        color   = self._cfg.colors.green if flash.passed else self._cfg.colors.red
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), color, -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, flash.text[:55], (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, self._cfg.colors.white, 2, cv2.LINE_AA)

    # ── All done overlay ───────────────────────────────────────────────────────

    def _draw_all_done(self, display: np.ndarray, w: int, h: int):
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 180, 60), -1)
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
        cv2.putText(display, "ALL STEPS COMPLETE!", (60, h // 2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, self._cfg.colors.white, 2, cv2.LINE_AA)
        cv2.putText(display, "Press R to reset", (160, h // 2 + 30),
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
