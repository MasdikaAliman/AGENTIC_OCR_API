import os
import time
import cv2
import queue
import threading
from config import load_config, AppConfig
from icecream import ic

# ── Verifier bootstrap ─────────────────────────────────────────────────────────

def build_verifiers(cfg: AppConfig) -> dict:
    """
    For every inspect-mode step, create a SOPVerifier backed by its own
    SOPReferenceBank loaded from the step's reference_folder.

    Returns a dict  { step_id: SOPVerifier }SOP_CUSTOM.yaml
    Only imports DINOv2 / torch if there are actually inspect steps (lazy load).
    """
    inspect_steps = [s for s in cfg.sop_steps if s.needs_inspect]
    if not inspect_steps:
        print("[INIT] No inspect steps — skipping DINOv2 load.")
        return {}

    from SOPReferenceBank import SOPReferenceBank
    from SOPVerifier import SOPVerifier

    enc_cfg = cfg.dino   # InspectEncoderConfig
    enc_name = enc_cfg.encoder.lower()

    if enc_name == "lightglue":
        from LightGlueEncoder import LightGlueEncoder
        encoder = LightGlueEncoder(
            features=enc_cfg.lightglue_features,
            max_num_keypoints=enc_cfg.max_num_keypoints,
            max_expected_inliers=enc_cfg.max_expected_inliers,
            depth_confidence=enc_cfg.depth_confidence,
            width_confidence=enc_cfg.width_confidence,
            filter_threshold=enc_cfg.filter_threshold,
        )
    elif enc_name == "xfeat":
        from XFeatEncoder import XFeatEncoder
        encoder = XFeatEncoder(
            top_k=enc_cfg.max_num_keypoints,
            detection_threshold=enc_cfg.detection_threshold,
            max_expected_inliers=enc_cfg.max_expected_inliers,
        )
    elif enc_name == "dinov2":
        from DINOv2Encoder import DINOv2Encoder
        encoder = DINOv2Encoder()
    elif enc_name == "cnn":
        from CNNEncoder import CNNEncoder
        encoder = CNNEncoder("resnet50")
    else:
        raise ValueError(
            f"Unknown encoder '{enc_name}' in config.yaml inspect.encoder. "
            f"Choose 'xfeat' or 'lightglue'."
        )
    print(f"[INIT] Encoder: {enc_name.upper()} ready.")
    verifiers: dict = {}

    for step in inspect_steps:
        ins = step.inspect
        threshold = ins.pass_threshold if ins.pass_threshold is not None \
                    else cfg.dino.pass_threshold

        print(f"  → Building reference bank for '{step.name}' "
              f"from '{ins.reference_folder}' (threshold={threshold:.2f})")

        bank = SOPReferenceBank(encoder)

        # Use cached embeddings if available — skip slow re-encoding
        embed_path = ins.reference_folder / "embeddings"
        meta_path  = str(embed_path) + "_metadata.json"
        npy_path   = str(embed_path) + "_embeddings.npy"

        if os.path.exists(meta_path) and os.path.exists(npy_path):
            print(f"    Loading cached embeddings from {embed_path}")
            bank.load(str(embed_path))
        else:
            print("Load from reference folder image")
            image_files = [
                f for f in os.listdir(str(ins.reference_folder))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if not image_files:
                print(f"    [WARN] No images found in '{ins.reference_folder}'. "
                      f"Step {step.name} inspect will always fail. "
                      f"Add reference images and restart.")
                continue
            bank.register_from_root(str(ins.reference_folder))
            bank.save(str(embed_path))

        verifiers[step.step_id] = SOPVerifier(
            encoder=encoder,
            bank=bank,
            pass_threshold=threshold,
        )

    print(f"[INIT] {len(verifiers)} verifier(s) ready.\n")
    return verifiers


# ── Main loop ──────────────────────────────────────────────────────────────────

class SOPPipeline:

    def __init__(self, cfg, engine, renderer, tracker):

        self.cfg = cfg
        self.engine = engine
        self.renderer = renderer
        self.tracker = tracker

        self.is_running = False

        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)

        self.capture_thread = None
        self.process_thread = None


    def _capture_loop(self, cap):

        while self.is_running:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1280, 720))

            # Drop old frame if queue full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.frame_queue.put(frame)


    def _process_loop(self):

        from hand_tracker import HandState

        while self.is_running:

            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

       
            if not self.engine.all_done:
                hand = self.tracker.process(
                    frame,
                    self.engine.current_step
                )
            else:
                hand = HandState()

            flash = self.engine.update(hand, frame=frame)

            result = {
                "frame": frame,
                "hand": hand,
                "flash": flash
            }


            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass

            self.result_queue.put(result)


    def run(self, cap):

        self.is_running = True

        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(cap,),
            daemon=True
        )

        self.process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True
        )

        self.capture_thread.start()
        self.process_thread.start()

        fps = 0
        prev_t = time.time()

        cv2.namedWindow("SOP Assembly")

        latest_result = None

        while cap.isOpened() and self.is_running:

            # Get newest processed result
            try:
                latest_result = self.result_queue.get_nowait()
            except queue.Empty:
                pass

            if latest_result is not None:

                display = latest_result["frame"].copy()

                fps = 0.9 * fps + 0.1 / max(time.time() - prev_t, 1e-6)
                prev_t = time.time()

                self.renderer.draw_frame(
                    display,
                    self.engine,
                    latest_result["hand"],
                    latest_result["flash"],
                    fps
                )

                cv2.imshow("SOP Assembly", display)

            key = cv2.waitKey(1) & 0xFF

            if key in [27, ord('q')]:
                self.is_running = False
                break
            elif key == ord("p"):
                print("Paused, press any key to continue")
                cv2.waitKey(0)
            elif key == ord('r'):
                self.engine.reset()

        self.is_running = False

        self.capture_thread.join(timeout=1.0)
        self.process_thread.join(timeout=1.0)

        cap.release()
        cv2.destroyAllWindows()

def run_sop_logic_zone():
    from hand_tracker import HandTracker
    from sop_engine import SOPEngine
    from renderer import Renderer

    cfg       = load_config("SOP_CUSTOM.yaml")
    verifiers = build_verifiers(cfg)
    engine    = SOPEngine(cfg, verifiers=verifiers)
    renderer  = Renderer(cfg)

    cap = cv2.VideoCapture(cfg.camera.source, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.camera.frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.frame_h)
    cap.set(cv2.CAP_PROP_FPS,          cfg.camera.fps)

    with HandTracker(cfg) as tracker:
        pipeline = SOPPipeline(cfg, engine, renderer, tracker)
        pipeline.run(cap)

    cap.release()
    cv2.destroyAllWindows()


# ── Offline: register reference images only ────────────────────────────────────

def register_references_only():
    """
    Utility: re-encode all inspect-step reference folders and save embeddings.
    Run this once after adding/changing reference images.
    """
    import shutil
    cfg = load_config("config.yaml")
    inspect_steps = [s for s in cfg.sop_steps if s.needs_inspect]

    if not inspect_steps:
        print("No inspect steps found in config.yaml.")
        return

    from DINOv2Encoder import DINOv2Encoder
    from SOPReferenceBank import SOPReferenceBank

    encoder = DINOv2Encoder(model_name=cfg.dino.model_name)

    for step in inspect_steps:
        ins = step.inspect
        embed_path = ins.reference_folder / "embeddings"
        print(f"\n[REGISTER] {step.name} → {ins.reference_folder}")
        bank = SOPReferenceBank(encoder)
        bank.register_from_folder(str(ins.reference_folder))
        bank.save(str(embed_path))
        print(f"  Saved to {embed_path}")

    print("\nDone. Run run_sop_logic_zone() to start the main pipeline.")


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_sop_logic_zone()
    # register_references_only()  # uncomment to re-encode reference images