from camera.capture_video import capture_video
from gesture_recognition.hand_tracking import HandTracker


def main():
    capture_video(hand_tracker)


if __name__ == "__main__":
    hand_tracker = HandTracker()
    main()
