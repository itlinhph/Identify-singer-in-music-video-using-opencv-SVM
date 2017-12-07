import numpy as np
import os, sys
import cv2
from PIL import Image
from skimage import io
from skimage import color

rawPath = '/media/linhphan/LEARN/CNTT 2/IS YEAR 4 SEMESTER 1/ML- KhoatTQ/BTL/rawdata/'
resizePath = '/media/linhphan//CNTT 2/IS YEAR 4 SEMESTER 1/ML- KhoatTQ/BTL/resizedata/'
dirs = ['1/']
# dirs = ['1/','2/','3/','4/','5/','6/','7/','8/']
imgDataFinal =[]

def resizeImg():
    faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    for dir in dirs:
        path = rawPath + dir
        for imgfile in os.listdir(path):
            if os.path.isfile(path+ imgfile):
                img = io.imread(path + imgfile)
                name, ext = os.path.splitext(imgfile)
                
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # faces = faceDetect.detectMultiScale(gray, 1.3, 5)
                faces = faceDetect.detectMultiScale(img)
                print(type(faces))
                print(path+name+ext)
                if (isinstance(faces, tuple) ):
                    print('delete!')
                    # print(faces.shape)
                    continue
                # print(faces)
                print(faces.shape)
                x,y = faces[0][:2]
                cropped = img[y: y+32, x: x+32]
                if(cropped.shape != (32, 32, 3)):
                    continue
                print(cropped.shape)
                imgDataFinal.append(cropped)
                # imgresize = cropped.resize((32,32))
                # cropped.save(resizePath+dir+name+ext)

resizeImg()
print(len(imgDataFinal))
dst = np.zeros([len(imgDataFinal), 32, 32, 3], dtype=float)
for i in range(0, len(imgDataFinal)):
   dst[i, :, :, :] = imgDataFinal[i]
print(type(imgDataFinal))
print(type(dst))

# print(imgDataFinal[0].shape)
print(dst[0, :, :, :])


