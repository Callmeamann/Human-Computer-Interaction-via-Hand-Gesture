import tensorflow as tf
import numpy as np


class KeyPointClassifier:
    def __init__(self, model_path='C:/Users/amang/MyProjects/Professional/HCI using CV/src/model_label/keypoint_classifier.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def __call__(self, landmark_list):
        input_tensor_index = self.interpreter.get_input_details()[0]['index']
        output_tensor_index = self.interpreter.get_output_details()[0]['index']

        input_data = np.array([landmark_list], dtype=np.float32)
        self.interpreter.set_tensor(input_tensor_index, input_data)
        self.interpreter.invoke()

        result = self.interpreter.get_tensor(output_tensor_index)
        result_index = np.argmax(np.squeeze(result))

        return result_index


