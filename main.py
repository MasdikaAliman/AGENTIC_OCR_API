import time

import cv2
from SOPVerifier import SOPVerifier
from DINOv2Encoder import  DINOv2Encoder
from SOPReferenceBank import SOPReferenceBank
def run_live(verifier: SOPVerifier, capture_key='s', quit_key='q'):
    """
    Press `s` to capture and verify the current frame.
    Press `q` to quit.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,2160)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)

    print(f"\nStarting. Expecting step {verifier.current_step}: "
          f"'{verifier.bank.steps[verifier.current_step]['step_name']}'")

    while True:
        ret, frame = cap.read()
        resize_frame = cv2.resize(frame.copy(), (640, 480))
        if not ret:
            break

        # Overlay current expected step on screen
        step_name = verifier.bank.steps[verifier.current_step]['step_name']
        cv2.putText(resize_frame, f"Expected: {step_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(resize_frame, f"Press '{capture_key}' to verify", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('SOP Verifier', resize_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(capture_key):
            cv2.imwrite(f"capture_{step_name}.png", frame.copy())
            result = verifier.verify(frame)
            color = (0, 255, 0) if result['passed'] else (0, 0, 255)
            print(f"\n{result['message']}")
            print(f"  Similarity: {result['similarity']}")

            # Flash result on screen for 1.5s
            overlay = resize_frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 80), color, -1)
            cv2.addWeighted(overlay, 0.4, resize_frame, 0.6, 0, resize_frame)
            cv2.putText(resize_frame, result['message'][:60], (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('SOP Verifier', resize_frame)
            cv2.waitKey(1500)

        elif key == ord(quit_key):
            break

    cap.release()
    cv2.destroyAllWindows()


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked: ({x}, {y})")

def run_sop_logic_zone():
    from config import load_config
    from hand_tracker import HandTracker, HandState
    from sop_engine import SOPEngine
    from renderer import Renderer

    cfg = load_config("config.yaml")
    engine = SOPEngine(cfg)
    renderer = Renderer(cfg)

    cap = cv2.VideoCapture(cfg.camera.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.frame_h)
    cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)

    fps, prev_t = 0.0, time.time()
    cv2.namedWindow("SOP Assembly")
    cv2.setMouseCallback("SOP Assembly", mouse_callback)

    with HandTracker(cfg) as tracker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            display = cv2.resize(frame, (cfg.camera.frame_w, cfg.camera.frame_h))

            # Hand detection (draws landmarks onto display)
            hand: HandState = HandState()
            if not engine.all_done:
                hand = tracker.process(frame, display, engine.current_step)

            # SOP state machine
            flash = engine.update(hand)

            #Render all overlays
            fps = 0.9 * fps + 0.1 / max(time.time() - prev_t, 1e-6)
            prev_t = time.time()
            renderer.draw_frame(display, engine, hand, flash, fps)

            cv2.imshow("SOP Assembly", display)

            # Hotkeys
            # if hand.in_wrong_zone:
            #     print("Wrong detected lamp on")
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                cv2.imwrite(f"snap_{int(time.time())}.png", display)
                print(f"[SNAP] Saved snapshot")
            elif key == ord("p") or hand.in_wrong_zone:
                print("Paused — press any key to continue")
                cv2.waitKey(0)
            elif key == ord("r"):
                engine.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_sop_logic_zone()
    # # 1. Init encoder
    # encoder = DINOv2Encoder(model_name='dinov2_vitb14')
    #
    # # 2. Build reference bank (run once)
    # bank = SOPReferenceBank(encoder)
    # bank.register_from_folder("image_test/SOP")
    # bank.save('image_test/SOP')
    #
    # # --- OR load existing bank ---
    # # bank = SOPReferenceBank(encoder)
    # # bank.load('image_test/SOP')
    #
    # # 3. Create verifier
    # verifier = SOPVerifier(
    #     encoder=encoder,
    #     bank=bank,
    #     pass_threshold=0.8   # tune this per your use case
    # )
    # #
    # # # 4. Run live
    # run_live(verifier)