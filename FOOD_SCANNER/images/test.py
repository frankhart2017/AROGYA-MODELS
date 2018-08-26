from keras.models import load_model
import cv2
import numpy as np
import os

model = load_model('vgg19_1.h5')

classes = {0: 'aloo beans', 1: 'apple', 2: 'banana', 3: 'besan adoo', 4: 'bhindi sabji'}

from PIL import Image
from matplotlib.pyplot import imshow

for image in images:
    pil_im = Image.open('test/' + image, 'r')
    imshow(np.asarray(pil_im))
    im = cv2.imread('test/' + image)
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (128, 128)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis =0)
    print('This is', classes[np.argmax(model.predict(im))])
