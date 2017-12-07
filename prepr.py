import numpy as np
import os, sys
import cv2
from PIL import Image
from skimage import io
from skimage import color

rawPath = 'rawdata/'
outputPath = 'output/'

dirs = ['1/']
# dirs = ['1/','2/','3/','4/','5/','6/','7/','8/']
imgDataFinal =[]

def faceDetectAndResizeImg():
    faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    for dir in dirs:
        path = rawPath + dir
        for imgfile in os.listdir(path):
            if os.path.isfile(path+ imgfile):
                img = io.imread(path + imgfile)
                name, ext = os.path.splitext(imgfile)
                # Detect Faces
                faces = faceDetect.detectMultiScale(img)
                print(path+name+ext)
                if (isinstance(faces, tuple) ):
                    print('Cannot Detect!')
                    continue
                x,y,w,h = faces[0][:4]
                faceCrop = img[y:y + h, x:x + w]
                croppedImg = cv2.resize(faceCrop, (32,32))
                cv2.imwrite(outputPath + dir + name + ext, croppedImg)

    print('Face recognize Complete!')


def faceToVetor():
    imgData =[]
    for dir in dirs:
        path = outputPath + dir
        for imgfile in os.listdir(path):
            if os.path.isfile(path + imgfile):
                img = io.imread(path + imgfile, as_grey=True)
                name, ext = os.path.splitext(imgfile)
                if(img.shape != (32, 32)):
                    print('Remove: '+ path+name+ext)
                    continue
                imgData.append(img)
    imgDataFinal = np.array(imgData)
    print(imgDataFinal.shape)
    print(imgDataFinal[0])
    train = imgDataFinal.reshape(len(imgDataFinal), 32*32)
    print(train.shape)
    print(train[0])


# faceDetectAndResizeImg()
faceToVetor()


# print(len(imgDataFinal))
# dst = np.zeros([len(imgDataFinal), 32, 32, 3], dtype=float)
# for i in range(0, len(imgDataFinal)):
#    dst[i, :, :, :] = imgDataFinal[i]
# print(type(imgDataFinal))
# print(type(dst))

# # print(imgDataFinal[0].shape)
# print(dst[0, :, :, :])


