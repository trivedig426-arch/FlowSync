import time
import pyautogui
import screen_brightness_control as sbc

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

def extract_coords(point):
    if point is None:
        return None, None
    if isinstance(point, dict):
        return point.get("x", None), point.get("y", None)
    if hasattr(point, "x") and hasattr(point, "y"):
        return point.x, point.y
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return point[0], point[1]
    return None, None

class GestureController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        
        self.last_action_time = {
            "click": 0,
            "right_click": 0,
            "double_click": 0,
            "volume": 0,
            "brightness": 0
        }
        
        self.is_dragging = False
        self.prev_scroll_y = None
        self.prev_mouse_x = self.screen_w // 2
        self.prev_mouse_y = self.screen_h // 2
        self.smooth_factor = 0.35

    def _get_pt(self, landmarks, key_name, index_num):
        if isinstance(landmarks, dict):
            return landmarks.get(key_name, landmarks.get(index_num, landmarks.get(str(index_num))))
        elif isinstance(landmarks, (list, tuple)) and len(landmarks) > index_num:
            return landmarks[index_num]
        return None

    def execute_action(self, gesture, landmarks, frame_w, frame_h):
        current_time = time.time()

        index_pt = self._get_pt(landmarks, "INDEX_FINGER_TIP", 8)
        ix, iy = extract_coords(index_pt)

        # 1. Mouse Cursor Move (Jab tak fist ya scroll na ho, cursor ungli ko follow karega)
        if ix is not None and iy is not None:
            # Margins: screen corners reach karne ke liye 0.15 to 0.85 map kiya hai
            norm_x = (ix - 0.15) / 0.70
            norm_y = (iy - 0.15) / 0.70
            
            target_x = max(10, min(self.screen_w - 10, int(norm_x * self.screen_w)))
            target_y = max(10, min(self.screen_h - 10, int(norm_y * self.screen_h)))

            cur_x = int(self.prev_mouse_x + (target_x - self.prev_mouse_x) * self.smooth_factor)
            cur_y = int(self.prev_mouse_y + (target_y - self.prev_mouse_y) * self.smooth_factor)

            pyautogui.moveTo(cur_x, cur_y)
            self.prev_mouse_x, self.prev_mouse_y = cur_x, cur_y

        # 2. Left Click
        if gesture == "index_pinch":
            if current_time - self.last_action_time["click"] > 0.35:
                pyautogui.click()
                self.last_action_time["click"] = current_time

        # 3. Right Click
        elif gesture == "middle_pinch":
            if current_time - self.last_action_time["right_click"] > 0.45:
                pyautogui.rightClick()
                self.last_action_time["right_click"] = current_time

        # 4. Double Click
        elif gesture == "double_pinch":
            if current_time - self.last_action_time["double_click"] > 0.55:
                pyautogui.doubleClick()
                self.last_action_time["double_click"] = current_time

        # 5. Drag & Drop (Fist)
        if gesture == "fist":
            if not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True
        else:
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False

        # 6. Scroll (Peace)
        if gesture == "peace" and iy is not None:
            if self.prev_scroll_y is not None:
                dy = iy - self.prev_scroll_y
                if abs(dy) > 0.01:
                    pyautogui.scroll(int(-dy * 500))
            self.prev_scroll_y = iy
        else:
            self.prev_scroll_y = None

        # 7. Volume
        if gesture == "thumb_up":
            if current_time - self.last_action_time["volume"] > 0.18:
                pyautogui.press("volumeup")
                self.last_action_time["volume"] = current_time

        elif gesture == "thumb_down":
            if current_time - self.last_action_time["volume"] > 0.18:
                pyautogui.press("volumedown")
                self.last_action_time["volume"] = current_time

        # 8. Brightness (C-shape)
        elif gesture == "c_shape" and iy is not None:
            if current_time - self.last_action_time["brightness"] > 0.25:
                try:
                    cur = sbc.get_brightness()
                    b = cur[0] if isinstance(cur, list) else cur
                    if iy < 0.35:
                        sbc.set_brightness(min(100, b + 5))
                    elif iy > 0.65:
                        sbc.set_brightness(max(5, b - 5))
                    self.last_action_time["brightness"] = current_time
                except Exception:
                    pass