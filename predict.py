import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

CATEGORIES = ['Cat', 'Dog']
test_photo = ['images/Cat (0).jpg', 'images/Cat (1).jpg', 'images/Cat (2).jpg', 'images/Cat (3).jpg',
              'images/Cat (4).jpg', 'images/Cat (5).jpg', 'images/Cat (6).jpg', 'images/Cat (7).jpg',
              'images/Cat (8).jpg', 'images/Cat (9).jpg']


def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Wrong path:', path)
    else:
        new_arr = cv2.resize(img, (60, 60))
        new_arr = np.array(new_arr)
        new_arr = new_arr.reshape(-1, 60, 60, 1)
        return new_arr


model = keras.models.load_model('cat-vs-dog.model')

# Переберём все фотографии из массива photo
for i in test_photo:
    prediction = model.predict([image(i)])
    print(i, ' : ', CATEGORIES[prediction.argmax()])
