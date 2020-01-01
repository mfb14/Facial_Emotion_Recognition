#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 14:22:51 2019

@author: furkanbilen
"""


import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import sys
import cv2
from keras_preprocessing import image


import matplotlib.pyplot as plt
from matplotlib import interactive


detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'
img_path = sys.argv[1]




img_orj = image.load_img(img_path)
img = image.load_img(img_path, grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = load_model(emotion_model_path).predict(x)
#emotion_analysis(custom[0])


#1
objects = ('kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal')
y_pos = np.arange(len(objects))
f = plt.figure(1) 
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel('yüzde')
plt.title('duygu')

#2
g = plt.figure(2)
x = np.array(x, 'float32')
x = x.reshape([48, 48]);
plt.axis('off')
plt.gray()
plt.imshow(img_orj)
plt.show()
#img2 = mpimg.imread(img_path)
#imgplot = plt.imshow(img2)
#plt.show()