import numpy as np
import os, sys
import cv2
from PIL import Image
from skimage import io

rawPath = 'rawdata/'
outputPath = 'output/'

# dirs = ['1/']
dirs = ['1/','2/','3/','4/','5/','6/','7/','8/']
imgDataFinal =[]
label = []

def faceDetectAndResizeImg():
    faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    for dir in dirs:
        path = rawPath + dir
        for imgfile in os.listdir(path):
            if os.path.isfile(path+ imgfile):
                img = cv2.imread(path + imgfile)
                name, ext = os.path.splitext(imgfile)
                # Detect Faces
                faces = faceDetect.detectMultiScale(img)
                # print(path+name+ext)
                if (isinstance(faces, tuple) ):
                    # print('Cannot Detect!')
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
                Y_label = dirs.index(dir)
                label.append(Y_label)
                
    imgDataFinal = np.array(imgData)
    Y_train = np.array(label)
    # print(imgDataFinal.shape)
    X_train = imgDataFinal.reshape(len(imgDataFinal), 32*32)
    X_train = np.round(X_train,4)
    print(X_train.shape)
    print(Y_train.shape)
    fileWrite = open('vector.txt', 'w')
    for i in range(0, len(imgDataFinal)):
        fileWrite.write(str(Y_train[i]) + ' ')
        for i2 in range(0,1024):
            fileWrite.write(str(i2)+':'+str(X_train[i][i2])+' ')

        fileWrite.write('\n')
    fileWrite.close()
    print('Save vector complete!')


# faceDetectAndResizeImg()
faceToVetor()





