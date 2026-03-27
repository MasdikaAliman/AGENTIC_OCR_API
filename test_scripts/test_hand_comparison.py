import cv2
import mediapipe as mp
import numpy as np
import time
from icecream import ic

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

FINGER_TIPS = [4, 8, 12, 16, 20]
THUMB_FINGER = [2, 3, 4]
INDEX_FINGER = [5, 6, 7, 8]
MIDDLE_FINGER = [9, 10, 11, 12]  # FIX 1: was never used — now actually used below

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
CLR_BG = (15, 15, 25)
CLR_ACCENT = (0, 210, 255)
CLR_GREEN = (50, 220, 90)
CLR_RED = (60, 60, 220)
CLR_WHITE = (240, 240, 240)
CLR_YELLOW = (30, 220, 200)
CLR_PANEL = (25, 25, 40)

SIMILARITY_THRESHOLD = 0.8  # tune: lower = looser match


# ── Embedding helpers ──────────────────────────────────────────────────────────

def extract_hand_data(lms, label: str) -> dict:
    """
    Build one clean hand dict from a single MediaPipe landmark result.
    FIX 1: lists are created INSIDE this function → no cross-hand accumulation.
    FIX 2: mid_finger now iterates MIDDLE_FINGER, not THUMB_FINGER.
    """
    wrist = (lms.landmark[0].x, lms.landmark[0].y, lms.landmark[0].z)

    thumb_data = [[lms.landmark[i].x, lms.landmark[i].y, lms.landmark[i].z]
                  for i in THUMB_FINGER]

    index_data = [[lms.landmark[i].x, lms.landmark[i].y, lms.landmark[i].z]
                  for i in INDEX_FINGER]

    # FIX 2: was `for mid_id in THUMB_FINGER` — should be MIDDLE_FINGER
    mid_data = [[lms.landmark[i].x, lms.landmark[i].y, lms.landmark[i].z]
                for i in MIDDLE_FINGER]

    return {
        "label": label,
        "wrist": wrist,
        "thumb_point": thumb_data,
        "index_point": index_data,
        "mid_point": mid_data,
    }


def parse_to_array(hand: dict) -> np.ndarray:
    """
    Flatten hand dict → (11, 2) float32 array.
    Order: wrist(1) + thumb(3) + index(4) + mid(3) = 11 pts  → but mid is now 4 pts
    so total = 1+3+4+4 = 12 pts → (12, 2) → 24-D vector.
    """
    pts = []
    wx, wy, wz = hand['wrist']
    pts.append([wx, wy, wz])
    for x, y, z in hand['thumb_point']:
        pts.append([x, y, z])
    for x, y, z in hand['index_point']:
        pts.append([x, y, z])
    for x, y, z in hand['mid_point']:
        pts.append([x, y, z])
    return np.array(pts, dtype=np.float32)


def embed(hand: dict) -> np.ndarray:
    """
    Normalised embedding vector from hand dict.
    1. Translate: subtract wrist so wrist = (0,0)
    2. Scale:     divide by max landmark distance from wrist
    3. Flatten:   (N, 2) → (2N,) float32
    """
    pts = parse_to_array(hand)
    # pts -= pts[0]  # translate to wrist origin
    # max_dist = np.max(np.linalg.norm(pts, axis=1))
    # if max_dist > 1e-6:
    #     pts /= max_dist  # unit scale
    # return pts.flatten()


    # ✅ center ke area jari (bukan wrist)
    center = (pts[5] + pts[9]) / 2
    pts -= center

    # ✅ scale pakai struktur tulang
    scale = np.linalg.norm(pts[5] - pts[9])
    if scale > 1e-6:
        pts /= scale

    # ✅ base feature
    base = pts.flatten()

    # ✅ tambahan penting (grasp detection)
    thumb_tip = pts[3]
    index_tip = pts[7]
    middle_tip = pts[11]

    d1 = np.linalg.norm(thumb_tip - index_tip)
    d2 = np.linalg.norm(index_tip - middle_tip)
    d3 = np.linalg.norm(thumb_tip - middle_tip)

    extra = np.array([d1, d2, d3])

    return np.concatenate([base, extra])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_landmark_ids(frame, hand_landmarks, w, h):
    for i, lm in enumerate(hand_landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.putText(frame, str(i), (cx - 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, CLR_WHITE, 1, cv2.LINE_AA)


def draw_sim_bar(frame, sim: float, x=14, y=60, w=200, h=14):
    """Draw a similarity score bar on the frame."""
    filled = int(w * max(0.0, sim))
    color = CLR_GREEN if sim >= SIMILARITY_THRESHOLD else CLR_ACCENT
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.putText(frame, f"sim: {sim:.3f}", (x + w + 8, y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_status(frame, text: str, color, y=100):
    cv2.rectangle(frame, (10, y - 18), (10 + len(text) * 9 + 6, y + 6),
                  (20, 20, 20), -1)
    cv2.putText(frame, text, (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

def is_hand_grip_from_lms(lms, threshold=0.25):
    pts = []
    for lm in lms.landmark:
        pts.append([lm.x, lm.y])
    pts = np.array(pts)

    # palm center
    center = (pts[0] + pts[5] + pts[17]) / 3

    tip_ids = [4, 8, 12, 16, 20]

    close_count = 0

    for tip in tip_ids:
        dist = np.linalg.norm(pts[tip] - center)
        if dist < threshold:
            close_count += 1

    return close_count >= 3


# ── Main ───────────────────────────────────────────────────────────────────────
def process_reference_hand_pose(file_path_name,handspose):
    image_referece = cv2.imread(file_path_name)
    rgb_frame=cv2.cvtColor(image_referece, cv2.COLOR_BGR2RGB)
    results  = handspose.process(rgb_frame)
    embedded_results = None
    if results.multi_hand_landmarks and results.multi_handedness:
            for lms, handedness in zip(results.multi_hand_landmarks,
                                       results.multi_handedness):
                label = handedness.classification[0].label

                # FIX 1 + FIX 2: use the clean extractor function
                hand = extract_hand_data(lms, label)
                embedded_results = embed(hand)
    return embedded_results

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"Camera: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
    print("Hotkeys:  R = capture reference   Q/ESC = quit")

    custom_conn_style = mp_drawing.DrawingSpec(color=CLR_ACCENT, thickness=2)
    custom_lm_style = mp_drawing.DrawingSpec(color=CLR_GREEN, thickness=4,
                                             circle_radius=4)

    hands_pose = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.5,
    )

    # reference_embedding = None  # set when user presses R
    fps = 0.0
    prev_t = time.time()
    sim_score = 0.0

    reference_embedding = process_reference_hand_pose("../image_test/SOP/STEP_2.png", hands_pose)

    ic(reference_embedding.shape)



    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Frame grab error")
            break

        frame = cv2.flip(frame, 1)
        visual_frame = frame.copy()
        h, w = visual_frame.shape[:2]

        rgb_frame = cv2.cvtColor(visual_frame, cv2.COLOR_BGR2RGB)
        results_handpose = hands_pose.process(rgb_frame)

        #Zone Items
        cv2.rectangle(visual_frame, (246, 2), (453, 186), CLR_RED, 2)
        cv2.rectangle(visual_frame, (365, 162), (494, 356), CLR_YELLOW, 2)
        cv2.rectangle(visual_frame, (149, 163), (298, 368), CLR_GREEN, 2)


        hand_data = []
        hand_in_area = False
        live_embedding = None
        is_grip = False
        if results_handpose.multi_hand_landmarks and results_handpose.multi_handedness:
            for lms, handedness in zip(results_handpose.multi_hand_landmarks,
                                       results_handpose.multi_handedness):
                label = handedness.classification[0].label

                # FIX 1 + FIX 2: use the clean extractor function
                hand = extract_hand_data(lms, label)
                hand_data.append(hand)

                # Compute live embedding from the first detected hand
                if live_embedding is None:
                    live_embedding = embed(hand)

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    visual_frame, lms, mp_hands.HAND_CONNECTIONS,
                    custom_lm_style, custom_conn_style,
                )
                draw_landmark_ids(visual_frame, lms, w, h)
                is_grip = is_hand_grip_from_lms(lms)

                if is_grip:
                    draw_status(visual_frame, "GRIP", CLR_GREEN, y=140)
                else:
                    draw_status(visual_frame, "OPEN", CLR_YELLOW, y=140)

                # Highlight fingertips
                for tip_id in FINGER_TIPS:
                    lm = lms.landmark[tip_id]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # ic(cx, cy, tip_id)
                    if not hand_in_area and  (149<= cx <=298) and (163 <= cy <=368):
                        hand_in_area = True
                    cv2.circle(visual_frame, (cx, cy), 10, CLR_YELLOW, -1)
                    cv2.circle(visual_frame, (cx, cy), 10, CLR_WHITE, 2)

        # ── Similarity comparison ──────────────────────────────────────────
        if reference_embedding is not None and live_embedding is not None:
            sim_score = cosine_similarity(reference_embedding, live_embedding)
            draw_sim_bar(visual_frame, sim_score)

            if (is_grip or sim_score >= SIMILARITY_THRESHOLD) and hand_in_area :
                draw_status(visual_frame, "MATCH", CLR_GREEN, y=100)
            else:
                draw_status(visual_frame, "NO MATCH", CLR_RED, y=100)

        elif reference_embedding is None:
            draw_status(visual_frame, "Press R to capture reference", CLR_ACCENT, y=100)
        else:
            draw_status(visual_frame, "No hand detected", (120, 120, 120), y=100)

        # ── Reference indicator ────────────────────────────────────────────
        if reference_embedding is not None:
            cv2.circle(visual_frame, (w - 20, 20), 8, CLR_GREEN, -1)
            cv2.putText(visual_frame, "REF", (w - 50, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, CLR_GREEN, 1, cv2.LINE_AA)

        # ── FPS ────────────────────────────────────────────────────────────
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t = now
        cv2.putText(visual_frame, f"FPS: {fps:.1f}", (14, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_GREEN, 1, cv2.LINE_AA)

        cv2.imshow("Visual", visual_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        if key == ord('s'):
            print("Save Pict")
            cv2.imwrite("_.png", frame.copy())

        # ── R: capture reference ───────────────────────────────────────────
        if key == ord('r'):
            if live_embedding is not None:
                reference_embedding = live_embedding.copy()
                ic(reference_embedding.shape)
                ic(reference_embedding)
                ic(hand_data)

                print("[INFO] Reference captured.")
            else:
                print("[WARN] No hand in frame — move hand into view first.")

    cap.release()
    cv2.destroyAllWindows()
    hands_pose.close()


if __name__ == "__main__":
    main()