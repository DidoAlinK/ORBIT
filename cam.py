import time                     # precise clock for timing and timestamps
from collections import deque   # fast-to-trim list for the FPS counter
from pathlib import Path        # builds file paths that work on any OS

import cv2                      # OpenCV: camera, drawing, windows
import mediapipe as mp          # MediaPipe: gives us mp.Image
from mediapipe.tasks import python as mp_python      # BaseOptions (where the model file is)
from mediapipe.tasks.python import vision            # Hand, Pose and Face models

CAMERA_INDEX = 0                # which camera to open
OUT_W, OUT_H = 640, 360         # every frame is shrunk to this size
FLIP = True                     # mirror the frame before tracking
SWAP_LABELS = True              # fixes Left/Right on this feed (our Phone Link finding)
POSE_EVERY = 2                  # run pose on every 2nd frame, reuse the result in between
FACE_EVERY = 1                  # NEW: run face on every frame; raise to 2 or 3 if fps drops
MODELS = Path(__file__).parent / "models"            # models folder next to this file
HAND_MODEL = MODELS / "hand_landmarker.task"
POSE_MODEL = MODELS / "pose_landmarker_lite.task"
FACE_MODEL = MODELS / "blaze_face_short_range.tflite"   # NEW
FINGERTIPS = (4, 8, 12, 16, 20) # landmark indices: thumb, index, middle, ring, pinky tips
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12               # pose landmark numbers
MIN_VISIBILITY = 0.5            # below this, a shoulder counts as "not detected"


def draw_hands(frame, result):
    h, w = frame.shape[:2]
    for landmarks, handed in zip(result.hand_landmarks, result.handedness):
        label = handed[0].category_name              # "Left" or "Right"
        if SWAP_LABELS:
            label = "Left" if label == "Right" else "Right"
        for i, lm in enumerate(landmarks):
            px, py = int(lm.x * w), int(lm.y * h)
            if i in FINGERTIPS:
                cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
        wx, wy = int(landmarks[0].x * w), int(landmarks[0].y * h)
        cv2.putText(frame, label, (wx, wy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def draw_shoulders(frame, result):
    h, w = frame.shape[:2]
    if result is None or not result.pose_landmarks:  # no result yet, or nobody found
        return
    pose = result.pose_landmarks[0]
    points = []
    for idx in (LEFT_SHOULDER, RIGHT_SHOULDER):
        lm = pose[idx]
        if lm.visibility < MIN_VISIBILITY:
            continue
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 8, (255, 0, 255), -1)
        cv2.putText(frame, f"{lm.visibility:.2f}", (px + 10, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    if len(points) == 2:
        cv2.line(frame, points[0], points[1], (255, 0, 255), 2)


def draw_face(frame, result):                        # NEW function
    if result is None:                               # no result yet (before the first run)
        return
    h, w = frame.shape[:2]
    for det in result.detections:                    # empty list = no face, loop runs zero times
        box = det.bounding_box                       # PIXELS here, not 0-1 fractions
        cv2.rectangle(frame,
                      (box.origin_x, box.origin_y),                         # top-left corner
                      (box.origin_x + box.width, box.origin_y + box.height),  # bottom-right corner
                      (255, 200, 0), 2)
        cv2.putText(frame, f"{det.categories[0].score:.2f}",                # confidence score
                    (box.origin_x, box.origin_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        for i, kp in enumerate(det.keypoints):       # 6 points, as 0-1 fractions
            px, py = int(kp.x * w), int(kp.y * h)
            color = (0, 255, 255) if i < 2 else (255, 200, 0)   # first two = eyes, in yellow
            cv2.circle(frame, (px, py), 4, color, -1)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)             # add cv2.CAP_DSHOW here if you needed it
    if not cap.isOpened():
        print("Error: could not open the camera.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)              # keep only the newest frame queued

    for path in (HAND_MODEL, POSE_MODEL, FACE_MODEL):    # check every model file exists
        if not path.exists():
            print(f"Error: model not found at {path}")
            cap.release()
            return

    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    ))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    ))
    face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(   # NEW
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,       # same video mode as the others
        min_detection_confidence=0.4,    # default is 0.5; weaker detections get dropped
    ))

    cv2.namedWindow("ORBIT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ORBIT", 1280, 720)

    stamps = deque()
    timings = {"capture": 0.0, "resize": 0.0, "hands": 0.0,
               "pose": 0.0, "face": 0.0, "draw": 0.0}                # "face" is NEW
    start = time.perf_counter()
    frame_count = 0                                  # counts frames for the every-Nth logic
    pose_result = None                               # last pose result, reused between runs
    face_result = None                               # NEW: last face result, reused between runs

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        timings["capture"] = (t1 - t0) * 1000
        if not ok:
            print("Error: lost the camera feed.")
            break

        t2 = time.perf_counter()
        frame = cv2.resize(frame, (OUT_W, OUT_H))
        if FLIP:
            frame = cv2.flip(frame, 1)
        timings["resize"] = (time.perf_counter() - t2) * 1000

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert once, all models reuse it
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - start) * 1000)

        t3 = time.perf_counter()
        hand_result = hands.detect_for_video(mp_image, timestamp_ms)
        timings["hands"] = (time.perf_counter() - t3) * 1000

        if frame_count % POSE_EVERY == 0:
            t3b = time.perf_counter()
            pose_result = pose.detect_for_video(mp_image, timestamp_ms)
            timings["pose"] = (time.perf_counter() - t3b) * 1000

        if frame_count % FACE_EVERY == 0:            # NEW block, same pattern as pose
            t3c = time.perf_counter()
            face_result = face.detect_for_video(mp_image, timestamp_ms)
            timings["face"] = (time.perf_counter() - t3c) * 1000

        frame_count += 1

        now = time.perf_counter()
        stamps.append(now)
        while stamps and now - stamps[0] > 1.0:
            stamps.popleft()
        fps = len(stamps)

        t4 = time.perf_counter()
        draw_hands(frame, hand_result)
        draw_shoulders(frame, pose_result)
        draw_face(frame, face_result)                # NEW
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y = 60
        for name, ms in timings.items():
            cv2.putText(frame, f"{name}: {ms:.1f} ms", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25

        cv2.imshow("ORBIT", frame)
        key = cv2.waitKey(1) & 0xFF
        timings["draw"] = (time.perf_counter() - t4) * 1000

        if key == ord("q"):
            break

    face.close()                                     # NEW: release models
    pose.close()
    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()