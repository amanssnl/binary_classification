#import necessary library

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
import numpy as np

model = tf.keras.models.load_model('vgg16.h5')  #load model for prediction



def predict(filename):
    image = load_img(filename, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    output = model.predict(img)
    # print(output)

    if output<0.4:
        return "Lion"
    else:
        return "Non_Lion"

    # print(model.summary)


str=predict('images/pig1.jpg')
print("prdicted image is ",str)

