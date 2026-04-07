"""
sop_engine.py — SOP state machine.
Knows nothing about drawing or camera; only advances state based on HandState.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time

import icecream

from config import AppConfig, SOPStep
from hand_tracker import HandState


# ── Per-step runtime state ─────────────────────────────────────────────────────

@dataclass
class StepRuntime:
    picked:      bool  = False
    picked_time: float = 0.0
    at_assembly: bool  = False
    grip_start: float = 0.0

# ── Flash message (returned to Renderer, keeps engine drawing-free) ────────────

@dataclass
class FlashMessage:
    passed:  bool
    text:    str


# ── SOPEngine ──────────────────────────────────────────────────────────────────

class SOPEngine:
    """
    Drives the SOP workflow.
    Call update(hand_state) every frame → may advance current_step or set all_done.
    """

    def __init__(self, cfg: AppConfig):
        self._cfg          = cfg
        self._steps        = cfg.sop_steps
        self._runtimes: list[StepRuntime] = [StepRuntime() for _ in self._steps]
        self.current_step  = 0
        self.all_done      = False
        self._session_start: float = 0.0   # set on first grip-in-zone
        self._session_end:   float = 0.0   # set when all_done

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def step_cfg(self) -> SOPStep:
        return self._steps[self.current_step]

    @property
    def runtime(self) -> StepRuntime:
        return self._runtimes[self.current_step]

    def update(self, hand: HandState) -> FlashMessage | None:
        """
        Advance state based on hand.
        Returns a FlashMessage to display, or None if nothing to flash.
        """
        if self.current_step == 0:
            self.start_time_sop = time.time()
        if self.all_done:
            self.reset()
            elapsed_all_step = time.time() - self.start_time_sop
            print(f"Total Time: {elapsed_all_step}")
            return None

        rt = self.runtime
        # icecream.ic(self._runtimes)
        # icecream.ic(rt.picked, rt.at_assembly)
        if not rt.picked:
            return self._handle_pre_pick(rt, hand)
        else:
            return self._handle_post_pick(rt, hand)

    def reset(self):
        self._runtimes  = [StepRuntime() for _ in self._steps]
        self.current_step = 0
        self.all_done     = False
        print("\n[RESET] Back to Step 1")

    # ── Private ────────────────────────────────────────────────────────────────

    def _handle_pre_pick(self, rt: StepRuntime, hand: HandState) -> FlashMessage | None:
        if rt.grip_start == 0.0:
            rt.grip_start = time.time()

        if hand.in_wrong_zone:
            print("WARNING ZONE")
            if self.current_step > 0:
                self.current_step -= 1
                self._runtimes[self.current_step].picked = False
                self._runtimes[self.current_step].at_assembly = False
            return FlashMessage(False, "WARNING: Wrong zone! Go to correct pick area.")

        if hand.in_pick and hand.grip:
            # Start the dwell timer on first frame of grip-in-zone
            elapsed = time.time() - rt.grip_start
            # Show progress so user knows to hold
            icecream.ic(elapsed)
            if elapsed < self._cfg.gesture.pick_dwell_time:
                pct = int((elapsed / self._cfg.gesture.pick_dwell_time) * 100)
                # icecream.ic(elapsed, self._cfg.gesture.pick_dwell_time)
                return FlashMessage(None, f"Hold grip... {pct}%")  # neutral flash

            # Dwell satisfied → confirm pick
            rt.picked = True
            rt.picked_time = time.time()
            print(f"[PICK] {self.step_cfg.name} — item picked!")

        else:
            # Grip broken or left zone — reset dwell timer
            rt.grip_start = 0.0

        return None

    def _handle_post_pick(self, rt: StepRuntime, hand: HandState) -> FlashMessage | None:
        if hand.in_assembly:
            rt.at_assembly = True

        if rt.at_assembly:
            elapsed = time.time() - rt.picked_time
            # icecream.ic(elapsed)
            if elapsed > self._cfg.gesture.success_delay:
                self._advance()
            return FlashMessage(True, f"{self.step_cfg.name} SUCCESS!")

        return FlashMessage(False, f"{self.step_cfg.name} — Bring to ASSEMBLY ZONE!")

    def _advance(self):
        if self.current_step < len(self._steps) - 1:
            self.current_step += 1
            print(f"\n[NEXT] Proceeding to {self.step_cfg.name}")
        else:
            self.all_done = True
            print("\n[DONE] All SOP steps completed!")
