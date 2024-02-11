import copy
import itertools
from src.gesture_recognition.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from src.gesture_recognition.point_history_classfier.point_history_classifier import PointHistoryClassifier


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


class GesturePredictionModule:
    def __init__(self):
        self.keypoint_classifier = KeyPointClassifier()
        # self.point_history_classifier = PointHistoryClassifier()

    def __call__(self, image, landmarks, keypoint_classifier_label):
        # Landmark calculation
        landmark_list = calc_landmark_list(image, landmarks)

        # Conversion to relative coordinates / normalized coordinates
        pre_processed_landmark_list = pre_process_landmark(landmark_list)

        # Predict Hand Sign index according to keypoint_classification_label
        hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

        # From keypoint_classification_label.csv match the index and return the label
        return keypoint_classifier_label[hand_sign_id], hand_sign_id
