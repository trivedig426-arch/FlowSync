import os
import sys
import cv2
import joblib
import numpy as np
import time
import math
import ctypes
import pyautogui
from collections import deque, Counter

user32 = ctypes.windll.user32
SCREEN_W = int(user32.GetSystemMetrics(0))
SCREEN_H = int(user32.GetSystemMetrics(1))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from Camera.camera import open_camera, get_frame
from hand_tracking.hand_tracker import HandTracker
from Landmarks.landmarks_processor import process_landmarks
from Landmarks.landmarks_smoothing import LandmarkSmoother
from features.cv_pipeline import get_hand_features

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

MODEL_PATH = os.path.join(BASE_DIR, "ML_Module", "gesture_model.pkl")

def get_pt(landmarks, key_name, idx):
    if landmarks is None:
        return None
    if isinstance(landmarks, dict):
        if key_name in landmarks and landmarks[key_name] is not None:
            pt = landmarks[key_name]
            return float(pt['x']), float(pt['y'])
        if idx in landmarks and landmarks[idx] is not None:
            pt = landmarks[idx]
            return float(pt['x']), float(pt['y'])
    if isinstance(landmarks, (list, tuple, np.ndarray)):
        if len(landmarks) > idx and landmarks[idx] is not None:
            pt = landmarks[idx]
            if isinstance(pt, (list, tuple, np.ndarray)):
                return float(pt[0]), float(pt[1])
            if isinstance(pt, dict):
                return float(pt.get('x', 0.0)), float(pt.get('y', 0.0))
            return float(getattr(pt, 'x', 0.0)), float(getattr(pt, 'y', 0.0))
    return None

def calc_dist(p1, p2):
    if p1 is None or p2 is None:
        return 999.0
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file missing: {MODEL_PATH}")
        return

    clf = joblib.load(MODEL_PATH)
    cap = open_camera()
    if cap is None:
        print("Camera nahi khula!")
        return

    tracker = HandTracker()
    smoother = LandmarkSmoother(alpha=0.70)
    gesture_buffer = deque(maxlen=5)

    is_dragging = False
    left_pinched_prev = False
    right_pinched_prev = False

    last_vol_t = 0
    last_shot_t = 0
    fist_time = 0

    win_name = "FlowSync Engine"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 480)

    print("\n--- FlowSync Running (Press 'q' ya ESC band karne ke liye) ---\n")

    try:
        while True:
            frame = get_frame(cap)
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            curr_t = time.time()

            result = tracker.detect(frame)
            processed = process_landmarks(result)

            hand_present = False
            pred_label = "none"
            raw_landmarks = None
            smoothed_landmarks = None

            if processed and len(processed) > 0:
                hand_present = True
                hand = processed[0]
                raw_landmarks = hand["landmarks"]

                feat_res = get_hand_features(raw_landmarks, hand["finger_status"])
                if feat_res and feat_res["valid"]:
                    feats = np.array(feat_res["feature_vector"], dtype=np.float32).reshape(1, -1)
                    probs = clf.predict_proba(feats)[0]
                    pred_label = clf.classes_[np.argmax(probs)]

                try:
                    smoothed_landmarks = smoother.smooth(raw_landmarks)
                except Exception:
                    smoothed_landmarks = raw_landmarks

            if hand_present:
                gesture_buffer.append(pred_label)
            else:
                gesture_buffer.clear()
                gesture_buffer.append("none")

            active_gesture = Counter(gesture_buffer).most_common(1)[0][0]

            if hand_present and (raw_landmarks is not None):
                target_lm = smoothed_landmarks if smoothed_landmarks is not None else raw_landmarks

                p0 = get_pt(target_lm, "Wrist", 0)
                p4 = get_pt(target_lm, "Thumb_Tip", 4)
                p8 = get_pt(target_lm, "Index_Tip", 8)
                p9 = get_pt(target_lm, "Middle_MCP", 9)
                p12 = get_pt(target_lm, "Middle_Tip", 12)

                if p8 is not None:
                    nx, ny = p8
                    cx = int(nx * w) if nx <= 1.0 else int(nx)
                    cy = int(ny * h) if ny <= 1.0 else int(ny)

                    # Green Dot
                    cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)
                    cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2)

                    # Screen cursor mapping
                    sx = int(np.interp(cx, [int(w * 0.15), int(w * 0.85)], [0, SCREEN_W]))
                    sy = int(np.interp(cy, [int(h * 0.15), int(h * 0.85)], [0, SCREEN_H]))
                    sx = max(0, min(SCREEN_W - 1, sx))
                    sy = max(0, min(SCREEN_H - 1, sy))

                    # Scale calculation
                    scale = calc_dist(p0, p9) if (p0 and p9) else 0.3
                    scale = 0.3 if scale < 0.05 else scale

                    dist_thumb_index = calc_dist(p4, p8) / scale
                    dist_thumb_middle = calc_dist(p4, p12) / scale

                    # Strict Pinch Distance: 0.13 se kam hone par hi physical contact count hoga
                    is_left_touch = dist_thumb_index < 0.13
                    is_right_touch = dist_thumb_middle < 0.13

                    # 1. Screenshot Logic
                    if active_gesture == "fist":
                        fist_time = curr_t

                    if active_gesture == "palm":
                        if 0.05 < (curr_t - fist_time) < 0.50 and (curr_t - last_shot_t) > 2.0 and fist_time != 0:
                            pyautogui.hotkey('win', 'prtsc')
                            last_shot_t = curr_t
                            fist_time = 0
                            print("[ACTION] Screenshot Captured!")
                        if is_dragging:
                            user32.mouse_event(0x0004, 0, 0, 0, 0)
                            is_dragging = False

                    # 2. Pointer Cursor Move
                    if active_gesture == "pointer" and not is_left_touch and not is_right_touch:
                        user32.SetCursorPos(sx, sy)
                        if is_dragging:
                            user32.mouse_event(0x0004, 0, 0, 0, 0)
                            is_dragging = False

                    # 3. Drag & Drop (Fist)
                    elif active_gesture == "fist":
                        user32.SetCursorPos(sx, sy)
                        if not is_dragging:
                            user32.mouse_event(0x0002, 0, 0, 0, 0)
                            is_dragging = True

                    # 4. Single-Shot Left Click (Sirf pinch create hone ke exact moment par ek click)
                    if is_left_touch and not left_pinched_prev and not is_dragging:
                        user32.mouse_event(0x0002, 0, 0, 0, 0)
                        user32.mouse_event(0x0004, 0, 0, 0, 0)
                        print("[ACTION] Left Click")
                    left_pinched_prev = is_left_touch

                    # 5. Single-Shot Right Click (Sirf pinch create hone ke exact moment par ek click)
                    if is_right_touch and not right_pinched_prev and not is_dragging:
                        user32.mouse_event(0x0008, 0, 0, 0, 0)
                        user32.mouse_event(0x0010, 0, 0, 0, 0)
                        print("[ACTION] Right Click")
                    right_pinched_prev = is_right_touch

                    # 6. Volume Control
                    if active_gesture == "thumb_up":
                        if curr_t - last_vol_t > 0.22:
                            pyautogui.press("volumeup")
                            last_vol_t = curr_t

                    elif active_gesture == "thumb_down":
                        if curr_t - last_vol_t > 0.22:
                            pyautogui.press("volumedown")
                            last_vol_t = curr_t
            else:
                left_pinched_prev = False
                right_pinched_prev = False
                if is_dragging:
                    user32.mouse_event(0x0004, 0, 0, 0, 0)
                    is_dragging = False

            display_status = active_gesture.upper() if hand_present else "NO HAND DETECTED"
            cv2.rectangle(frame, (10, 10), (450, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"STATUS: {display_status}", (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow(win_name, frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break

    finally:
        if is_dragging:
            user32.mouse_event(0x0004, 0, 0, 0, 0)
        cap.release()
        cv2.destroyAllWindows()
        print("FlowSync Engine Cleanly Closed.")

if __name__ == "__main__":
    main()