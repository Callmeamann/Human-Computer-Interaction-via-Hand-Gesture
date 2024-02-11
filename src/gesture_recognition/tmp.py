import cv2
import mediapipe as mp
from src.gesture_recognition.gesture_prediction import GesturePredictionModule
from src.utils.cv_fps_calc import CvFpsCalc
cvFpsCalc = CvFpsCalc(buffer_len=10)


class HandTracker:
    def __init__(self, static_mode=False, max_hands=1, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_mode, max_hands, model_complexity, min_detection_confidence, min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_detector = GesturePredictionModule()

    def find_hands(self, frame, keypoint_classifier_label, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        # Display FPS on the window
        fps = cvFpsCalc.get()
        fps_text = "FPS:" + str(fps)
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (211, 211, 211), 2, cv2.LINE_AA)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Get bounding box coordinates
                    bbox = self.calculate_bounding_box(frame, hand_landmarks)

                    # Gesture Prediction
                    gesture = self.gesture_detector(frame,hand_landmarks,keypoint_classifier_label)

                    # Add Gesture Label to Frame
                    cv2.putText(frame, f"Gesture: {gesture}", (bbox[0][0], bbox[0][1] - 30), cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 255, 255), 4)
                    cv2.putText(frame, f"Gesture: {gesture}", (bbox[0][0], bbox[0][1] - 30), cv2.FONT_HERSHEY_COMPLEX,0.5, (0, 0, 0), 2)

                    # Draw bounding box
                    cv2.rectangle(frame, bbox[0], bbox[1], (128, 128, 128), 4)
                    cv2.rectangle(frame, bbox[0], bbox[1], (0, 0, 0), 2)

                    # Add label for each hand
                    cv2.putText(frame, f"Hand {results.multi_hand_landmarks.index(hand_landmarks) + 1}",(bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 4)
                    cv2.putText(frame, f"Hand {results.multi_hand_landmarks.index(hand_landmarks) + 1}",(bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

        return frame, results

    def calculate_bounding_box(self, frame, hand_landmarks):
        h, w, c = frame.shape

        landmarks = []
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks.append((x, y))

        min_x = max(0, min(landmarks, key=lambda x: x[0])[0] - 10)
        min_y = max(0, min(landmarks, key=lambda x: x[1])[1] - 10)
        max_x = min(w, max(landmarks, key=lambda x: x[0])[0] + 10)
        max_y = min(h, max(landmarks, key=lambda x: x[1])[1] + 10)

        return (min_x, min_y), (max_x, max_y)
