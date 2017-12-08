import numpy as np
import os, sys
import cv2
from PIL import Image
from skimage import io
from sklearn import svm

rawPath = 'rawdata/'
outputPath = 'output/'

# dirs = ['1/']
imgDataFinal =[]

def faceDetectAndResizeImg(inputDir):
    faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    print('STARTING DETECT AND RESIZE FACE...\n')
    listDirs = os.listdir(inputDir)
    for subDir in listDirs:
        path = inputDir + subDir + '/'
        for imgfile in os.listdir(path):
            if os.path.isfile(path+ imgfile):
                img = cv2.imread(path + imgfile)
                # Detect Faces
                faces = faceDetect.detectMultiScale(img)
                print(path + imgfile)
                if (isinstance(faces, tuple) ):
                    print('FALSE: ' + path + imgfile)
                    continue
                x,y,w,h = faces[0][:4]
                faceCrop = img[y:y + h, x:x + w]
                croppedImg = cv2.resize(faceCrop, (32,32))
                if not os.path.exists(outputPath+subDir):
                    os.makedirs(outputPath + subDir)
                cv2.imwrite(outputPath + subDir +'/'+ imgfile , croppedImg)

    print('Face recognize Complete!')


def faceToVetor(inputDir):
    listImgFace =[]
    listLabel = []
    listDirs = os.listdir(inputDir)
    for subDir in listDirs:
        path = inputDir + subDir + '/'
        for imgfile in os.listdir(path):
            if os.path.isfile(path + imgfile):
                img = io.imread(path + imgfile, as_grey=True)
                if(img.shape != (32, 32)):
                    print('Remove: '+ path+imgfile)
                    continue
                listImgFace.append(img)
                Y_label = listDirs.index(subDir)
                listLabel.append(Y_label)
                
    imgDataFinal = np.array(listImgFace)
    Y_train = np.array(listLabel)
    X_train = imgDataFinal.reshape(len(imgDataFinal), 32*32)
    # X_train = np.round(X_train,5)
    print(X_train.shape)
    print(Y_train.shape)

    return X_train, Y_train
    
def writeVector():
    X_train, Y_train = faceToVetor(outputPath)
    fileWrite = open('vector.txt', 'w')
    for i in range(0, len(Y_train)):
        fileWrite.write(str(Y_train[i]) + ' ')
        for i2 in range(0, 1024):
            fileWrite.write(str(i2) + ':' + str(X_train[i][i2]) + ' ')

        fileWrite.write('\n')
    fileWrite.close()
    print('Save vector complete!')


def train():
    X_train, Y_train = faceToVetor(outputPath)
    X_test, Y_test = faceToVetor('test/')
    clf = svm.SVC()
    print(clf.fit(X_train, Y_train))
    
    print('train complete!')
    Y_pre = clf.predict(X_test)
    print(clf.support_vectors_)

    print('Y Test:')
    print(Y_test)
    print('Y pre:')
    print(Y_pre)
    

# faceDetectAndResizeImg(rawPath)
# faceToVetor(outputPath)
# writeVector()
train()





