import numpy as np
import os
import sys
import cv2
from PIL import Image
from skimage import io
from skimage import color

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = io.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, 1.3,5)
x,y,w,h = faces[0][:4]
cropped = img[y:y + h, x:x + w]

crop = cv2.resize(cropped,(32,32))
print(cropped.shape)
print(faces)

cv2.imwrite('reta4.jpg', cropped)
cv2.imwrite('crop.jpg', crop)
