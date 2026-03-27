
import cv2
import mediapipe as mp
import numpy as np
import time
from icecream import ic


# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ── Landmark groups ───────────────────────────────────────────────────────────
FINGER_TIPS   = [4, 8, 12, 16, 20]
FINGER_MCP   = [ 5, 9, 13]


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
CLR_WHITE  = (240, 240, 240)
CLR_GREEN  = (50,  220,  90)
CLR_RED    = (60,   60, 220)
CLR_YELLOW = (30,  220, 200)
CLR_ACCENT = (0,   210, 255)
CLR_ORANGE = (0,   140, 255)
CLR_GRAY   = (120, 120, 120)
CLR_PURPLE = (200,  80, 200)

# ── Tuning ────────────────────────────────────────────────────────────────────
GRIP_THRESHOLD  = 0.25   # normalized dist finger-tip → palm center
SUCCESS_DELAY   = 1.0    # Jeda waktu (detik) sebelum pindah ke step berikutnya

# ══════════════════════════════════════════════════════════════════════════════
#  SOP STEP CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SOP_STEPS = [
    {
        "step_id"     : 0,
        "name"        : "STEP 1",
        "instruction" : "Ambil item dari area MERAH",
        "zone_pick"   : (246,  2, 453, 186),
        "clr_pick"    : CLR_RED,
    },
    {
        "step_id"     : 1,
        "name"        : "STEP 2",
        "instruction" : "Ambil item dari area ORANGE",
        "zone_pick"   : (365, 162, 494, 356),
        "clr_pick"    : CLR_ORANGE,
    },
    {
        "step_id"     : 2,
        "name"        : "STEP 3",
        "instruction" : "Ambil item dari area UNGU",
        "zone_pick"   : (149, 163,298, 368),
        "clr_pick"    : CLR_PURPLE,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  HAND HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def is_grip(lms, threshold=GRIP_THRESHOLD) -> bool:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark])

    hand_scale = np.linalg.norm(pts[0] - pts[9])
    if hand_scale == 0:
        return False

    center = (pts[0] + pts[5] + pts[17]) / 3
    close_finger = 0

    for tip in FINGER_TIPS[1:]:  # skip thumb (index 4)
        dist = np.linalg.norm(pts[tip] - center)
        # normalized = dist/hand_scale
        # ic(normalized)
        if dist < threshold:  # normalized!
            close_finger += 1

    return close_finger >= 3

def fingertip_pixels(lms, w, h):
    return [(int(lms.landmark[t].x * w), int(lms.landmark[t].y * h)) for t in FINGER_TIPS]

def fingermcp_pixels(lms, w, h):
    return [(int(lms.landmark[t].x * w), int(lms.landmark[t].y * h)) for t in FINGER_MCP]

def point_in_zone(px, py, zone) -> bool:
    x1, y1, x2, y2 = zone
    return x1 <= px <= x2 and y1 <= py <= y2

def any_tip_in_zone(tips, zone) -> bool:
    return any(point_in_zone(cx, cy, zone) for cx, cy in tips)

# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_zone(frame, zone, color, label, active=False):
    x1, y1, x2, y2 = zone
    thickness = 3 if active else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.putText(frame, label, (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

def draw_label(frame, text, color, x=14, y=100):
    tw = len(text) * 9 + 8
    cv2.rectangle(frame, (x - 2, y - 18), (x + tw, y + 6), (15, 15, 25), -1)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

def draw_step_progress(frame, steps, current_step, w):
    panel_x = w - 220
    for i, step in enumerate(steps):
        y = 20 + i * 28
        if i < current_step:
            color, symbol = CLR_GREEN,  "DONE"
        elif i == current_step:
            color, symbol = CLR_YELLOW, "NOW >"
        else:
            color, symbol = CLR_GRAY,   "LOCK"
        cv2.putText(frame, f"{step['name']} [{symbol}]", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

def flash_result(frame, passed: bool, message: str):
    color   = CLR_GREEN if passed else CLR_RED
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), color, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, message[:55], (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, CLR_WHITE, 2, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════════════
#  PER-STEP RUNTIME STATE
# ══════════════════════════════════════════════════════════════════════════════

def make_step_runtime():
    return {
        "picked"     : False,
        "picked_time": None # Untuk tracking jeda transisi
    }

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Init MediaPipe ────────────────────────────────────────────────────────
    hands_model = mp_hands.Hands(
        # static_image_mode=True,
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.50,
    )
    conn_style = mp_drawing.DrawingSpec(color=CLR_ACCENT, thickness=2)
    lm_style   = mp_drawing.DrawingSpec(color=CLR_GREEN,  thickness=4, circle_radius=4)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── Runtime per step ──────────────────────────────────────────────────────
    runtimes     = [make_step_runtime() for _ in SOP_STEPS]
    current_step = 0
    all_done     = False

    fps      = 0.0
    prev_t   = time.time()

    print("\nHotkeys: Q/ESC = quit   S = snapshot   R = reset")
    print(f"Starting at {SOP_STEPS[0]['name']}: {SOP_STEPS[0]['instruction']}\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame        = cv2.flip(frame, 1)
        display      = frame.copy()
        h, w         = display.shape[:2]
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result    = hands_model.process(rgb)

        step_cfg     = SOP_STEPS[current_step] if not all_done else None
        rt           = runtimes[current_step] if not all_done else None

        # ── Draw all zones ────────────────────────────────────────────────────
        for i, s in enumerate(SOP_STEPS):
            # Hanya gambar area PICK
            zc_pick = s["clr_pick"] if i == current_step else CLR_GRAY
            label_pick = f"PICK S{i+1}" if i >= current_step else f"S{i+1} DONE"
            draw_zone(display, s["zone_pick"], zc_pick, label_pick,
                      active=(i == current_step and not (rt and rt["picked"])))

        # ── Progress panel ────────────────────────────────────────────────────
        draw_step_progress(display, SOP_STEPS, current_step, w)

        hand_in_pick  = False
        grip_detected = False

        if mp_result.multi_hand_landmarks and not all_done:
            for lms in mp_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display, lms, mp_hands.HAND_CONNECTIONS, lm_style, conn_style)

                tips = fingertip_pixels(lms, w, h)
                mcps = fingermcp_pixels(lms, w, h)
                for cx, cy in tips:
                    cv2.circle(display, (cx, cy), 9, CLR_YELLOW, -1)
                    cv2.circle(display, (cx, cy), 9, CLR_WHITE,  2)

                if any_tip_in_zone(mcps, step_cfg["zone_pick"]):
                    hand_in_pick = True

                if is_grip(lms):
                    grip_detected = True

        # ── Grip status label ─────────────────────────────────────────────────
        if mp_result.multi_hand_landmarks and not all_done:
            grip_text  = "GRIP" if grip_detected else "OPEN"
            grip_color = CLR_GREEN if grip_detected else CLR_YELLOW
            draw_label(display, grip_text, grip_color, x=14, y=140)

        # ══════════════════════════════════════════════════════════════════════
        #  LOGIC: PICK -> TRANSITION -> NEXT STEP
        # ══════════════════════════════════════════════════════════════════════
        if not all_done:
            # 1. Cek apakah tangan sudah ambil barang
            if not rt["picked"]:
                if hand_in_pick and grip_detected:
                    rt["picked"] = True
                    rt["picked_time"] = time.time()
                    print(f"[PICK] {step_cfg['name']} — item picked!")

            # 2. Jika sudah diambil, tunggu jeda waktu lalu lanjut
            else:
                flash_result(display, True, f"{step_cfg['name']} SUCCESS!")

                if time.time() - rt["picked_time"] > SUCCESS_DELAY:
                    # Advance ke step berikutnya
                    if current_step < len(SOP_STEPS) - 1:
                        current_step += 1
                        print(f"\n[NEXT] Proceeding to {SOP_STEPS[current_step]['name']}")
                    else:
                        all_done = True
                        print("\n[DONE] All SOP steps completed!")

        # ══════════════════════════════════════════════════════════════════════
        #  ALL DONE overlay
        # ══════════════════════════════════════════════════════════════════════
        if all_done:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 180, 60), -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            cv2.putText(display, "ALL STEPS COMPLETE!", (60, h // 2 - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, CLR_WHITE, 2, cv2.LINE_AA)
            cv2.putText(display, "Press R to reset", (160, h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_YELLOW, 1, cv2.LINE_AA)
            runtimes     = [make_step_runtime() for _ in SOP_STEPS]
            current_step = 0
            all_done     = False
        # ── HUD ───────────────────────────────────────────────────────────────
        inst_text = "ALL DONE — press R to reset" if all_done else step_cfg["instruction"]
        cv2.rectangle(display, (0, h - 30), (w, h), (15, 15, 25), -1)
        cv2.putText(display, inst_text, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_WHITE, 1, cv2.LINE_AA)

        if not all_done and not rt["picked"]:
            draw_label(display, f"[{step_cfg['name']}] Go to PICK zone + grip", CLR_ACCENT, x=14, y=100)

        # FPS
        now   = time.time()
        fps   = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(display, f"FPS {fps:.1f}", (14, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, CLR_GREEN, 1, cv2.LINE_AA)

        cv2.imshow("SOP Assembly", display)

        # ── Hotkeys ───────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27): break
        elif key == ord("s"): cv2.imwrite(f"snap_{int(time.time())}.png", frame)
        elif key == ord("r"):
            runtimes     = [make_step_runtime() for _ in SOP_STEPS]
            current_step = 0
            all_done     = False
            print("\n[RESET] Back to Step 1")

    cap.release()
    cv2.destroyAllWindows()
    hands_model.close()

if __name__ == "__main__":
    main()