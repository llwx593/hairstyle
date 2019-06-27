from __future__ import print_function

import keras
from keras.models import load_model
import numpy as np 
from skimage import io

from get_face_points import get_rotated_points_array

model1 = load_model("model/faceshape_classify_model_man.h5")
model2 = load_model("model/faceshape_classify_model_woman.h5")
FACE_NAME=['鹅蛋','方脸','心形','圆脸','长脸','钻石']

def predict_faceshape(user_gender, image_array):
    points_array = get_rotated_points_array(image_array)
    p = points_array.reshape(34,)
    np.set_printoptions(precision=4)
    if(user_gender ==0 ):
        predicted = model1.predict(p)
        return FACE_NAME[np.argmax(predicted)]
    else:
        predicted = model2.predict(p)
        return FACE_NAME[np.argmax(predicted)]

if __name__=='__main__':
    img = io.imread('test_images/22.jpg')
    #points_array = get_rotated_points_array(img)
    # io.imshow(img)
    # io.show()
    print(predict_faceshape(0,img))
