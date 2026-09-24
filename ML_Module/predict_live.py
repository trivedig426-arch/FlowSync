import cv2
import joblib
import numpy as np
import time
import math
import pyautogui
from collections import deque, Counter
import screen_brightness_control as sbc

from Camera.camera import open_camera, get_frame
from hand_tracking.hand_tracker import HandTracker
from Landmarks.landmarks_processor import process_landmarks
from Landmarks.landmarks_smoothing import LandmarkSmoother
from features.cv_pipeline import get_hand_features

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

SCREEN_W, SCREEN_H = pyautogui.size()
MODEL_PATH = "ML_Module/gesture_model.pkl"
CONFIDENCE_THRESHOLD = 0.55
BUFFER_SIZE = 4

def extract_pt(pt):
    if pt is None:
        return 0.0, 0.0
    if isinstance(pt, dict):
        return pt.get('x', 0.0), pt.get('y', 0.0)
    return getattr(pt, 'x', 0.0), getattr(pt, 'y', 0.0)

def get_dist(p1, p2):
    x1, y1 = extract_pt(p1)
    x2, y2 = extract_pt(p2)
    return math.hypot(x1 - x2, y1 - y2)

def refine_gesture(predicted_label, landmarks):
    try:
        thumb_tip = landmarks.get("THUMB_TIP", landmarks.get(4)) if isinstance(landmarks, dict) else landmarks[4]
        index_tip = landmarks.get("INDEX_FINGER_TIP", landmarks.get(8)) if isinstance(landmarks, dict) else landmarks[8]
        middle_tip = landmarks.get("MIDDLE_FINGER_TIP", landmarks.get(12)) if isinstance(landmarks, dict) else landmarks[12]
        wrist = landmarks.get("WRIST", landmarks.get(0)) if isinstance(landmarks, dict) else landmarks[0]
        mid_mcp = landmarks.get("MIDDLE_FINGER_MCP", landmarks.get(9)) if isinstance(landmarks, dict) else landmarks[9]

        if None in [thumb_tip, index_tip, middle_tip, wrist, mid_mcp]:
            return predicted_label

        scale = get_dist(wrist, mid_mcp) or 1.0
        d_thumb_index = get_dist(thumb_tip, index_tip) / scale
        d_thumb_middle = get_dist(thumb_tip, middle_tip) / scale

        # Pinches (Clicks)
        if d_thumb_index < 0.35 and d_thumb_index < d_thumb_middle:
            return "index_pinch"
        if d_thumb_middle < 0.35 and d_thumb_middle < d_thumb_index:
            return "middle_pinch"

        # Thumb direction check taaki random poses me Volume spam na ho
        _, ty = extract_pt(thumb_tip)
        _, wy = extract_pt(wrist)

        if predicted_label == "thumb_up" and ty >= wy:
            return "idle"
        if predicted_label == "thumb_down" and ty <= wy:
            return "idle"

    except Exception:
        pass
    return predicted_label

def run():
    clf = joblib.load(MODEL_PATH)
    cap = open_camera()
    if cap is None:
        print("Camera nahi mila!")
        return

    tracker = HandTracker()
    smoother = LandmarkSmoother(alpha=0.60)

    gesture_buffer = deque(maxlen=BUFFER_SIZE)
    prev_mx, prev_my = SCREEN_W // 2, SCREEN_H // 2
    smooth = 0.50

    is_dragging = False
    last_act = {"click": 0, "rclick": 0, "vol": 0, "bright": 0}
    prev_scroll_y = None

    print("\n>>> FLOWSYNC RUNNING: Press ESC on window to stop <<<\n")

    while cap.isOpened():
        frame = get_frame(cap)
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        result = tracker.detect(frame)
        processed_landmarks = process_landmarks(result)

        raw_gesture = "idle"
        raw_conf = 0.0
        hand_found = False
        target_pt = None

        for hand in processed_landmarks:
            hand_found = True
            smoothed = smoother.smooth(hand["landmarks"])
            output = get_hand_features(smoothed, hand["finger_status"])

            if output is None or not output["valid"]:
                continue

            features = np.array(output["feature_vector"]).reshape(1, -1)
            probs = clf.predict_proba(features)[0]
            max_idx = np.argmax(probs)
            pred_label = clf.classes_[max_idx]
            raw_conf = probs[max_idx]

            if raw_conf >= CONFIDENCE_THRESHOLD:
                raw_gesture = refine_gesture(pred_label, smoothed)
            else:
                raw_gesture = "idle"

            # Pointer tracking point
            if raw_gesture == "fist":
                target_pt = smoothed.get("MIDDLE_FINGER_MCP", smoothed.get(9)) if isinstance(smoothed, dict) else smoothed[9]
            else:
                target_pt = smoothed.get("INDEX_FINGER_TIP", smoothed.get(8)) if isinstance(smoothed, dict) else smoothed[8]

        gesture_buffer.append(raw_gesture if hand_found else "idle")
        stable_gesture = Counter(gesture_buffer).most_common(1)[0][0]

        curr_t = time.time()

        # OS ACTIONS
        if hand_found and target_pt is not None:
            px, py = extract_pt(target_pt)
            if px > 1.0 or py > 1.0:
                px, py = px / w, py / h

            norm_x = (px - 0.15) / 0.70
            norm_y = (py - 0.15) / 0.70
            target_x = max(5, min(SCREEN_W - 5, int(norm_x * SCREEN_W)))
            target_y = max(5, min(SCREEN_H - 5, int(norm_y * SCREEN_H)))

            cur_x = int(prev_mx + (target_x - prev_mx) * smooth)
            cur_y = int(prev_my + (target_y - prev_my) * smooth)

            # 1. Cursor Move (hamesha chalega, scroll me freeze)
            if stable_gesture != "peace":
                pyautogui.moveTo(cur_x, cur_y)
                prev_mx, prev_my = cur_x, cur_y

            # 2. Drag & Drop (Fist)
            if stable_gesture == "fist":
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                    print("[ACTION] Dragging Started")
            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                    print("[ACTION] Dropped")

            # 3. Left Click (Index Pinch)
            if stable_gesture == "index_pinch" and not is_dragging:
                if curr_t - last_act["click"] > 0.40:
                    pyautogui.click()
                    last_act["click"] = curr_t
                    print("[ACTION] Left Click")

            # 4. Right Click (Middle Pinch)
            elif stable_gesture == "middle_pinch" and not is_dragging:
                if curr_t - last_act["rclick"] > 0.50:
                    pyautogui.rightClick()
                    last_act["rclick"] = curr_t
                    print("[ACTION] Right Click")

            # 5. Scroll (Peace)
            elif stable_gesture == "peace":
                if prev_scroll_y is not None:
                    dy = py - prev_scroll_y
                    if abs(dy) > 0.015:
                        pyautogui.scroll(int(-dy * 1400))
                prev_scroll_y = py
            else:
                prev_scroll_y = None

            # 6. Volume Control (Cooldown badha diya taaki spam na ho)
            if stable_gesture == "thumb_up":
                if curr_t - last_act["vol"] > 0.35:
                    pyautogui.press("volumeup")
                    last_act["vol"] = curr_t
                    print("[ACTION] Volume Up")
            elif stable_gesture == "thumb_down":
                if curr_t - last_act["vol"] > 0.35:
                    pyautogui.press("volumedown")
                    last_act["vol"] = curr_t
                    print("[ACTION] Volume Down")

            # Visual Green/Red Tracking Point
            cv2.circle(frame, (int(px * w), int(py * h)), 10, (0, 0, 255) if is_dragging else (0, 255, 0), -1)

        else:
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False

        # Visual Dashboard Overlay
        badge_c = (0, 0, 255) if is_dragging else (0, 255, 0)
        cv2.rectangle(frame, (10, 10), (450, 75), (30, 30, 30), -1)
        txt = f"GESTURE: {stable_gesture.upper()}"
        if is_dragging:
            txt += " [DRAG]"
        cv2.putText(frame, txt, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, badge_c, 2)
        cv2.putText(frame, f"Conf: {raw_conf*100:.1f}%", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("FlowSync Controller", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    if is_dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()