import numpy as np
import cv2
import pickle
from tensorflow import keras
import pandas as pd

#from FrontEnd import select_image
#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.5  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRAINED MODEL

model = keras.models.load_model("model.h5")


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def equalize(img):
    img = cv2.equalizeHist(img)

    return img


def preprocessing(img):
    img = grayscale(img)


    img = equalize(img)
    img = img / 255
    return img


def getClassName(classNo):
    if classNo.all() == 0:
        return 'Cat'
    elif classNo.all() == 1:
        return 'Dog'

