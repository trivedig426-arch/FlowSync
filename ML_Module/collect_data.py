import os
import sys
import cv2
import time
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Camera.camera import open_camera, get_frame
from hand_tracking.hand_tracker import HandTracker
from Landmarks.landmarks_processor import process_landmarks
from Landmarks.landmarks_smoothing import LandmarkSmoother
from features.cv_pipeline import get_hand_features

CSV_PATH = os.path.join(BASE_DIR, "ML_Module", "dataset.csv")

GESTURES = [
    "palm",        # 1. Poora panja khula (Screen freeze/clutch)
    "pointer",     # 2. Sirf Index finger khuli (Cursor navigation)
    "left_click",  # 3. Index + Thumb tap
    "right_click", # 4. Middle + Thumb tap
    "fist",        # 5. Poori mutthi band (Drag / Hold)
    "thumb_up",    # 6. Angootha upar (Vol UP)
    "thumb_down"   # 7. Angootha neeche (Vol DOWN)
]

FRAMES_PER_GESTURE = 300

def get_landmark_coords(lm_data, target_idx=8):
    if lm_data is None:
        return None
    val = None
    if isinstance(lm_data, (list, tuple)):
        if len(lm_data) > target_idx:
            val = lm_data[target_idx]
    elif isinstance(lm_data, dict):
        if target_idx in lm_data:
            val = lm_data[target_idx]
        elif str(target_idx) in lm_data:
            val = lm_data[str(target_idx)]
        else:
            for k, v in lm_data.items():
                if "INDEX" in str(k).upper() and "TIP" in str(k).upper():
                    val = v
                    break
    if val is None:
        return None
    try:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return float(val[0]), float(val[1])
        if isinstance(val, dict):
            return float(val.get('x', 0.0)), float(val.get('y', 0.0))
        return float(getattr(val, 'x', 0.0)), float(getattr(val, 'y', 0.0))
    except Exception:
        return None

def main():
    cap = open_camera()
    if cap is None:
        print("Camera open nahi hua!")
        return

    tracker = HandTracker()
    smoother = LandmarkSmoother(alpha=0.70)

    dataset_rows = []

    print("\n==============================================")
    print("      FLOWSYNC DATA COLLECTION PIPELINE      ")
    print("==============================================")
    print(f"Total Gestures: {len(GESTURES)}")
    print(f"Frames per Gesture: {FRAMES_PER_GESTURE}")
    print("Press 'SPACE' to start recording current gesture.")
    print("Press 'Q' to abort.")
    print("==============================================\n")

    for g_idx, gesture in enumerate(GESTURES):
        print(f"\n[{g_idx+1}/{len(GESTURES)}] Target: >>> {gesture.upper()} <<<")
        print("Apna haath position par lao, aur tayar hone par SPACE dabao...")

        # Wait for user to be ready
        while True:
            frame = get_frame(cap)
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)

            cv2.rectangle(frame, (10, 10), (620, 60), (30, 30, 30), -1)
            cv2.putText(frame, f"NEXT: {gesture.upper()} (Press SPACE to record)", 
                        (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("FlowSync Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key in [27, ord('q'), ord('Q')]:
                cap.release()
                cv2.destroyAllWindows()
                print("Data collection cancelled.")
                return

        # 3-Second Countdown
        for sec in range(3, 0, -1):
            t_end = time.time() + 1.0
            while time.time() < t_end:
                frame = get_frame(cap)
                if frame is None:
                    continue
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Starting in: {sec}", (200, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                cv2.imshow("FlowSync Data Collector", frame)
                cv2.waitKey(1)

        # Record Frames
        collected = 0
        while collected < FRAMES_PER_GESTURE:
            frame = get_frame(cap)
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            result = tracker.detect(frame)
            processed_landmarks = process_landmarks(result)

            status_text = f"Recording: {gesture.upper()} ({collected}/{FRAMES_PER_GESTURE})"
            status_color = (0, 255, 0)

            if processed_landmarks and len(processed_landmarks) > 0:
                hand = processed_landmarks[0]
                smoothed = smoother.smooth(hand["landmarks"])
                output = get_hand_features(smoothed, hand["finger_status"])

                if output and output["valid"]:
                    row = list(output["feature_vector"])
                    row.append(gesture)
                    dataset_rows.append(row)
                    collected += 1

                    # Safe dot render
                    coords = get_landmark_coords(smoothed, 8)
                    if coords:
                        cx = int(coords[0] * w) if coords[0] <= 1.0 else int(coords[0])
                        cy = int(coords[1] * h) if coords[1] <= 1.0 else int(coords[1])
                        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                else:
                    status_color = (0, 165, 255)
                    status_text = "Hand features invalid! Adjust hand."
            else:
                status_color = (0, 0, 255)
                status_text = "No Hand Detected!"

            cv2.rectangle(frame, (10, 10), (550, 60), (30, 30, 30), -1)
            cv2.putText(frame, status_text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.imshow("FlowSync Data Collector", frame)

            if (cv2.waitKey(1) & 0xFF) in [27, ord('q'), ord('Q')]:
                break

    # Save to CSV
    if dataset_rows:
        num_features = len(dataset_rows[0]) - 1
        col_names = [f"f_{i}" for i in range(num_features)] + ["label"]
        df = pd.DataFrame(dataset_rows, columns=col_names)
        df.to_csv(CSV_PATH, index=False)
        print(f"\nSUCCESS: {len(df)} total samples saved cleanly to {CSV_PATH}")
    else:
        print("No samples recorded.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()