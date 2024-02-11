import math
import pyautogui
import cv2
import numpy as np
import mediapipe as mp
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from src.gesture_recognition.gesture_prediction import GesturePredictionModule
from src.utils.cv_fps_calc import CvFpsCalc

cvFpsCalc = CvFpsCalc(buffer_len=10)


def calculate_bounding_box(frame, hand_landmarks):
    h, w, _ = frame.shape

    # Extract (x, y) coordinates of landmarks and convert to pixel coordinates
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    # Calculate bounding box coordinates with a margin of 10 pixels
    min_x, min_y = max(0, min(landmarks, key=lambda x: x[0])[0] - 10), max(0, min(landmarks, key=lambda x: x[1])[1] - 10)
    max_x, max_y = min(w, max(landmarks, key=lambda x: x[0])[0] + 10), min(h, max(landmarks, key=lambda x: x[1])[1] + 10)

    return (min_x, min_y), (max_x, max_y)


def calculate_position(landmark, width, height):
    x, y = int(landmark.x * width), int(landmark.y * height)
    return x, y


class HandTracker:
    def __init__(self, static_mode=False, max_hands=1, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_mode, max_hands, model_complexity, min_detection_confidence, min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_detector = GesturePredictionModule()
        self.mode_label = "Tracker"
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.scale = 0.5
        self.thickness = 1
        self.color = (255, 255, 255)
        self.volume_control_active = False
        self.cursor_control_active = False
        self.screen_width, self.screen_height = pyautogui.size()
        self.active_id = 0
        self.volume = 1
        self.volume_previous = -1
        self.volume_change_count = 0
        self.system_volume_range = (-144.0, 0.0)
        self.initialize_volume_control()

    def initialize_volume_control(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = interface.QueryInterface(IAudioEndpointVolume)

    def find_hands(self, frame, keypoint_classifier_label, mode, frame_height, frame_width, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Display FPS on the window
        fps = cvFpsCalc.get()
        fps_text = "FPS:" + str(fps)
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (211, 211, 211), 2, cv2.LINE_AA)

        # Mode Selection
        if mode == 0:
            self.mode_label = "Tracker"
        elif mode == 1:
            self.volume_control_active = False
            self.cursor_control_active = False
            self.mode_label = "Gesture"
        else:
            self.mode_label = "Interactive"

        cv2.putText(frame, f"Mode : {self.mode_label}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"Mode : {self.mode_label}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1.0, (211, 211, 211), 2, cv2.LINE_AA)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Get bounding box coordinates
                    bbox = calculate_bounding_box(frame, hand_landmarks)

                    # Add label for each hand
                    cv2.putText(frame, f"Hand {results.multi_hand_landmarks.index(hand_landmarks) + 1}",
                                (bbox[0][0], bbox[0][1] - 10), self.font, self.scale, (0, 0, 0), self.thickness + 4,
                                cv2.LINE_AA)
                    cv2.putText(frame, f"Hand {results.multi_hand_landmarks.index(hand_landmarks) + 1}",
                                (bbox[0][0], bbox[0][1] - 10), self.font, self.scale, self.color, self.thickness,
                                cv2.LINE_AA)

                    # Draw bounding box
                    cv2.rectangle(frame, bbox[0], bbox[1], (128, 128, 128), 4)
                    cv2.rectangle(frame, bbox[0], bbox[1], (0, 0, 0), 2)

                    if mode == 0:
                        pass
                    else:
                        # Gesture Prediction
                        gesture = 'NULL'
                        if not (self.cursor_control_active or self.volume_control_active):
                            gesture, hand_sign_id = self.gesture_detector(frame, hand_landmarks, keypoint_classifier_label)
                            self.active_id = hand_sign_id
                        if mode == 1:
                            self.gesture_mode(frame, bbox, gesture)
                        else:
                            self.interactive_mode(frame, hand_landmarks, self.active_id, frame_height, frame_width)

        return frame

# Mode Function definitions
    def gesture_mode(self, frame, bbox, gesture):
        # Add text
        cv2.putText(frame, f"Gesture: {gesture}", (bbox[0][0], bbox[0][1] - 30), self.font, self.scale,
                    (0, 0, 0), self.thickness + 4, cv2.LINE_AA)
        cv2.putText(frame, f"Gesture: {gesture}", (bbox[0][0], bbox[0][1] - 30), self.font, self.scale,
                    self.color, self.thickness, cv2.LINE_AA)

    def interactive_mode(self, frame, hand_landmarks, hand_sign_id, frame_height, frame_width):
        if hand_sign_id == 4:
            # Volume control Flag ( gesture = V / two )
            self.volume_control(frame, hand_landmarks,  frame_height, frame_width)
        elif hand_sign_id == 7:
            # Cursor control Flag ( gesture = C / open mouth )
            self.cursor_control(hand_landmarks)
        elif hand_sign_id == 1:
            # Minimize the selected window
            pyautogui.hotkey('win', 'down')
            pyautogui.sleep(1.5)
        elif hand_sign_id == 5:
            # Switch tabs
            pyautogui.hotkey('alt', 'shift', 'tab')
            pyautogui.sleep(1.5)

# Helper Function Definitions

    # Volume related functions
    def volume_control(self, frame, hand_landmarks, frame_height, frame_width):
        if not self.volume_control_active:
            self.volume_control_active = True

        cursor_x, cursor_y = calculate_position(hand_landmarks.landmark[8], frame_width, frame_height)
        thumb_x, thumb_y = calculate_position(hand_landmarks.landmark[4], frame_width, frame_height)

        # Display points and line between them
        cv2.circle(frame, (cursor_x, cursor_y), 7, (113, 204, 46), cv2.FILLED)
        cv2.circle(frame, (thumb_x, thumb_y), 7, (113, 204, 46), cv2.FILLED)
        cv2.line(frame, (cursor_x, cursor_y), (thumb_x, thumb_y), (128, 195, 63), 3)

        # Calculate Volume
        volume_length = int(math.hypot(cursor_x - thumb_x, cursor_y - thumb_y))
        volume_length = max(20, min(170, volume_length))

        # Count the frames with diff of volume lower than or equal to threshold
        if abs(self.volume_previous - volume_length) < 5:
            self.volume_change_count += 1
        else:
            self.volume_change_count = 0
        self.volume_previous = volume_length

        # Convert Volume Length according to the System Volume range
        volume_percentage = int(np.interp(volume_length, [20, 170], [0, 100]))

        # Display volume text on the frame
        volume_text = f"Volume: {volume_percentage}%"
        cv2.putText(frame, volume_text, (10, 110), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, volume_text, (10, 110), cv2.FONT_HERSHEY_COMPLEX, 1.0, (211, 211, 211), 2, cv2.LINE_AA)

        # Convert Volume Length according to the System Volume range
        volume_length = int(
            np.interp(volume_length, [20, 170], [self.system_volume_range[0], self.system_volume_range[1]]))
        self.volume.SetMasterVolumeLevel(volume_length, None)

        # Check Exit condition ( hold your desired volume for 1-2 seconds )
        if self.volume_exit():
            self.volume_previous = -1
            self.volume_control_active = False
            self.volume_change_count = 0
            pyautogui.sleep(2)

    def volume_exit(self):
        return self.volume_change_count >= 30

    # Cursor related functions
    def cursor_control(self, hand_landmarks):
        if not self.cursor_control_active:
            self.cursor_control_active = True

        # Calculate cursor and thumb positions
        cursor_x, cursor_y = calculate_position(hand_landmarks.landmark[8], self.screen_width, self.screen_height)
        thumb_x, thumb_y = calculate_position(hand_landmarks.landmark[4], self.screen_width, self.screen_height)

        # Safe Cursor position
        margin = 10
        cursor_x = max(min(cursor_x, self.screen_width - margin), margin)
        cursor_y = max(min(cursor_y, self.screen_height - margin), margin)

        # Cursor movement
        pyautogui.moveTo(cursor_x, cursor_y)

        # Distance between Index finger and Thumb
        distance = int(math.hypot(cursor_x - thumb_x, cursor_y - thumb_y))
        # print(distance)

        # Click if conditions are met
        if distance < 50:
            pyautogui.click()
            pyautogui.sleep(1)
        # Exit Cursor Control (Widen the gap between the index finger and thumb)
        elif distance > 270:
            self.cursor_control_active = False
