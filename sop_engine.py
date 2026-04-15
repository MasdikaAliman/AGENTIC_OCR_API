
from __future__ import annotations

from typing import Optional

import hand_tracker
from dataclasses import dataclass, field
import time

from icecream import ic

from config import AppConfig, SOPStep
from hand_tracker import HandState
from hand_tracker  import _centroid_in_zone
from SOPVerifier import SOPVerifier
_INSPECT_COOLDOWN = 0.1
from FeatureBasedVerifier import FeatureBasedVerifier
# ── Per-step runtime state ─────────────────────────────────────────────────────

@dataclass
class StepRuntime:
    picked:              bool  = False
    picked_time:         float = 0.0
    at_assembly:         bool  = False
    grip_start:          float = 0.0
    # Inspect-specific
    inspect_passed:      bool  = False   # True once DINOv2 confirms the pick
    inspect_result:      Optional[dict] = None   # last raw verifier result
    inspect_last_run:    float = 0.0     # timestamp of last DINOv2 call (throttle)




@dataclass
class FlashMessage:
    passed:  bool
    text:    str


# ── SOPEngine ──────────────────────────────────────────────────────────────────

class SOPEngine:


    def __init__(self, cfg: AppConfig, verifiers: Optional[dict[int, "SOPVerifier"]] = None,
                 feature_verifiers: Optional[dict[int, "FeatureBasedVerifier"]] = None
                 ):
        self._cfg          = cfg
        self._steps        = cfg.sop_steps
        self._verifiers: dict[int, "SOPVerifier"] = verifiers or {}
        # self._verifiers: dict[int, "FeatureBasedVerifier"] = feature_verifiers or {} # Add feature verifier dict
        self._runtimes: list[StepRuntime] = [StepRuntime() for _ in self._steps]
        self.current_step  = 0
        self.all_done      = False
        self._start_time:    float = 0.0
        self._elapsed_total: float = 0.0

    @property
    def step_cfg(self) -> SOPStep:
        return self._steps[self.current_step]

    @property
    def runtime(self) -> StepRuntime:
        return self._runtimes[self.current_step]

    def update(self, hand: HandState, frame=None) -> FlashMessage | None:

        # if self.all_done:
        #     self.reset()
        #     return None

        # rt = self._runtimes[current]
        rt = self._runtimes[self.current_step]
        # icecream.ic(hand.auto_hand_step)
        # rt = self._runtimes[hand.auto_hand_step] # auto define step based on current_step
        # self.current_step = hand.auto_hand_step
        # self.current_step = current
        # icecream.ic(current)
        # icecream.ic(self._runtimes)
        # icecream.ic(rt.picked, rt.at_assembly)
        if not rt.picked:
            return self._handle_pre_pick(rt, hand)
        elif self.step_cfg.needs_inspect and not rt.inspect_passed and rt.picked:
                return self._handle_inspect(rt, frame)
        else:
            return self._handle_post_pick(rt, hand, frame)

    def reset(self):
        self._runtimes  = [StepRuntime() for _ in self._steps]
        self.current_step = 0
        self.all_done     = False
        for v in self._verifiers.values():
            v.reset()
        print("\n[RESET] Back to Step 1")

    # ── Private ────────────────────────────────────────────────────────────────

    def _handle_pre_pick(self, rt: StepRuntime, hand: HandState) -> FlashMessage | None:


        if hand.in_wrong_zone:
            rt.grip_start = 0.0
            return FlashMessage(False, "WARNING: Wrong zone! Go to correct pick area.")

        if hand.in_pick and hand.grip:
            now = time.time()
            if self._start_time == 0.0:
                self._start_time = now

            if rt.grip_start == 0.0:
                rt.grip_start = now

            elapsed = now - rt.grip_start
            # Show progress so user knows to hold
            # icecream.ic(elapsed)
            if elapsed < self._cfg.gesture.pick_dwell_time:
                pct = int((elapsed / self._cfg.gesture.pick_dwell_time) * 100)
                # icecream.ic(elapsed, self._cfg.gesture.pick_dwell_time)
                return FlashMessage(None, f"Hold grip... {pct}%")  # neutral flash

            # Dwell satisfied → confirm pick
            rt.picked = True
            rt.picked_time = now
            print(f"[PICK] {self.step_cfg.name} — item picked!")

        else:
            # Grip broken or left zone — reset dwell timer
            rt.grip_start = 0.0

        return None

    def _handle_post_pick(self, rt: StepRuntime, hand: HandState,
                          frame) -> FlashMessage | None:
        """
        After pick is confirmed:
          • hand_only : wait for assembly zone → advance
          • inspect   : run DINOv2 first, then wait for assembly zone → advance
        """
        step = self.step_cfg

        # ── Inspect gate (runs once until it passes) ───────────────────────

        ic(hand.hands_distance)
        # ── Assembly phase (same for both modes after inspect clears) ──────
        if hand.in_assembly and 10 <= hand.hands_distance <= 200:

            rt.at_assembly = True

        if rt.at_assembly or (step.inspect and rt.inspect_passed):
            # elapsed = time.time() - rt.picked_time
            # if elapsed > self._cfg.gesture.success_delay:
            self._advance()
            return FlashMessage(True, f"{step.name} SUCCESS!")

        return FlashMessage(False, f"{step.name} — Bring to ASSEMBLY ZONE!")

    def _handle_inspect(self, rt: StepRuntime, frame) -> FlashMessage | None:
        """
        Crop the current frame and run the DINOv2 verifier.
        Throttled by _INSPECT_COOLDOWN so we don't hammer CPU every frame.
        """
        step = self.step_cfg

        if frame is None:
            print(f"[WARN] Step {step.name} is inspect mode but no frame was passed "
                  f"to engine.update(). Skipping DINOv2 check.")
            rt.inspect_passed = True
            return None

        verifier = self._verifiers.get(step.step_id)
        if verifier is None:
            print(f"[WARN] No verifier registered for step_id={step.step_id}. "
                  f"Skipping DINOv2 check.")
            rt.inspect_passed = True
            return None

        # Throttle: only run inference every _INSPECT_COOLDOWN seconds
        now = time.time()
        if now - rt.inspect_last_run < _INSPECT_COOLDOWN:
            # Return the last known result message while waiting
            if rt.inspect_result is None:
                return FlashMessage(None, f"[{step.name}] Analysing item...")
            sim = rt.inspect_result["similarity"]
            thr = verifier.threshold
            # thr = verifier.match_ratio_threshold
            return FlashMessage(False, f"[INSPECT] sim={sim:.2f} / need {thr:.2f} — retrying...")

        rt.inspect_last_run = now

        # Crop the frame according to inspect config
        ins = step.inspect
        y1, y2, x1, x2 = ins.crop_coords
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            print(f"[WARN] Crop region is empty for step {step.name}. "
                  f"Check crop_coords in config.yaml.")
            rt.inspect_passed = True
            return None

        # Run verifier
        result = verifier.verify(crop)
        # result = verifier.verify(crop, expected_step_id=self.current_step)
        rt.inspect_result = result

        sim = result["similarity"]
        passed = result["passed"]

        if passed:
            rt.inspect_passed = True
            rt.picked_time = now  # reset so assembly phase timer is fresh
            print(f"[INSPECT] {step.name} | sim={sim:.3f} | passed={passed}")
            return FlashMessage(True, f"[INSPECT PASS] {step.name} — sim={sim:.2f}")
        else:
            # best = result.get("best_match_name", "?")
            # # best_sim = result.get("best_match_sim", 0.0)
            # # if (result.get("best_match_step") != verifier.
            # #         and best_sim >= verifier.pass_threshold):
            # #     msg = f"[INSPECT] Wrong item! Detected '{best}', expected '{name}'"
            # # else:
            # msg = (f"[INSPECT] Not recognised — "
            #            f"sim={sim:.2f} (need {verifier.pass_threshold:.2f})")
            msg = (f"[INSPECT] Not recognised — "
                       f"sim={sim:.2f} (need {verifier.threshold:.2f})")
            return FlashMessage(False, msg)

    def _advance(self):
        if self.current_step < len(self._steps) - 1:
            self.current_step += 1
            print(f"\n[NEXT] Proceeding to {self.step_cfg.name}")
        else:
            self.all_done = True
            print("\n[DONE] All SOP steps completed!")
