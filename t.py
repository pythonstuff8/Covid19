import numpy as np
# from google.colab.patches import cv2_imshow

import tensorflow as tf
from keras.preprocessing import image
def run_tflite_model(tflite_file, test_image):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])
    return predictions
model = tf.lite.Interpreter(model_path='model.tflite')
xtest_image = image.image_utils.load_img(r"C:\Users\stheMiner\Downloads\diease dection.v1i.folder\train\COVID19\COVID19-0-_jpg.rf.afa26aeea3a8ee5c60d1e75640e5357c.jpg", target_size = (299, 299))
xtest_image = image.image_utils.img_to_array(xtest_image)
test_image = np.expand_dims(xtest_image, axis = 0)
prediction = run_tflite_model('model.tflite', test_image)
print(prediction)
