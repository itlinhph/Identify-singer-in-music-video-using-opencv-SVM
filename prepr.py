import numpy as np
import os, sys
import cv2
from PIL import Image
from skimage import io
from sklearn import svm
from sklearn.svm import NuSVC

rawPath = 'rawdata/'
outputPath = 'output/'

def faceDetectAndResizeImg(inRawPath, outFacePath):
    faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    print('\n---STARTING DETECT AND RESIZE FACE---')
    listDirs = os.listdir(inRawPath)
    for subDir in listDirs:
        path = inRawPath + subDir + '/'
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
                if not os.path.exists(outFacePath+subDir):
                    os.makedirs(outFacePath + subDir)
                cv2.imwrite(outFacePath + subDir +'/'+ imgfile , croppedImg)

    print('---FACE REGCONIZE COMPLETE!---\n')


def faceToVetor(inFacePath):
    listImgFace =[]
    listLabel = []
    listDirs = os.listdir(inFacePath)
    for subDir in listDirs:
        path = inFacePath + subDir + '/'
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
    
def writeVector(faceDetectedPath, fileVectorPath):
    X_train, Y_train = faceToVetor(faceDetectedPath)
    fileWrite = open(fileVectorPath, 'w')
    for i in range(0, len(Y_train)):
        fileWrite.write(str(Y_train[i]) + ' ')
        for i2 in range(0, 1024):
            fileWrite.write(str(i2) + ':' + str(X_train[i][i2]) + ' ')

        fileWrite.write('\n')
    fileWrite.close()
    print('Save vector complete!')

def testASample(imgFile)

def train(trainPath, testPath):
    X_train, Y_train = faceToVetor(trainPath)

    faceDetectAndResizeImg(testPath, 'testFace/')
    X_test, Y_test = faceToVetor('testFace/')
    # clf = svm.SVC()
    clf = NuSVC()
    print(clf.fit(X_train, Y_train))
    
    print('---TRAIN COMPLETE!---\n')
    Y_pre = clf.predict(X_test)
    print(clf.support_vectors_)

    print('Y Test:')
    print(Y_test)
    print('Y pre:')
    print(Y_pre)
    

# faceDetectAndResizeImg('rawdata/', 'output/')
# faceToVetor('output/', )
# writeVector('output/', 'vector.txt')
train('output/', 'test/')





