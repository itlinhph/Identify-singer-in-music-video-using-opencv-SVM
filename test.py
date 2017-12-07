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


dst = np.zeros([len(ar), 32, 32, 3], dtype=float)
    for i in range(0, len(ar)):
        ar[i] = cv2.resize(ar[i], (32, 32), 0, 0)
        dst[i, :, :, :] = ar[i]

def applyGrayscaleAndEqualizeHist(data):
    length = len(data)
    data = data.astype(np.float, copy=False)
    print("Applying Grayscale filter and Histogram Equalization")

    filteredData = []

    for data_sample in data[0:length, :]:
        data_sample = np.float32(data_sample)
        grayScale = cv2.cvtColor(data_sample, cv2.COLOR_BGR2GRAY)
        grayScale = np.uint8(grayScale)
        equalized = cv2.equalizeHist(grayScale)
        filteredData.append(np.reshape(equalized, (32, 32, 1)))

    return np.array(filteredData)


def normalize(data):
    length = len(data)
    data = data.astype(np.float, copy=False)

    print("Starting normalization: ", datetime.datetime.now().time())
    for data_sample in data[0:length, :]:
        for data_sample_row in data_sample:
            for data_sample_pixel in data_sample_row:
                data_sample_pixel[:] = [
                    (color - 127.5) / 255.0 for color in data_sample_pixel]

    print("Normalization finished: ", datetime.datetime.now().time())
    return data
