import cv2
import csv


def capture_video(hand_tracker):

    def select_mode(key, mode):
        if key == ord('t'):
            return 0
        elif key == ord('g'):
            return 1
        elif key == ord('i'):
            return 2
        else:
            return mode

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define the window name
    window_name = "I m watching U"

    # Load the gesture labels from the csv file
    with open('C:/Users/amang/MyProjects/Professional/HCI using CV/src/model_label/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
        # for label in keypoint_classifier_labels:
        #     print(label,sep=" ")
    mode = 0
    while True:
        # Break the loop on 'q' key press
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        mode = select_mode(key, mode)

        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            print("Error: Could not read frame.")
            break

        # Horizontally flip the frame
        frame = cv2.flip(frame, 1)

        # Resize the Frame
        frame = cv2.resize(frame, (int(frame.shape[1] * 1.25), int(frame.shape[0] * 1.25)))
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        # Perform hand tracking
        frame = hand_tracker.find_hands(frame, keypoint_classifier_labels, mode, frame_height, frame_width)

        # Display the frame
        cv2.imshow(window_name, frame)

    # Release the camera
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
