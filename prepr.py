import numpy as np
import os, sys
import cv2
from PIL import Image
from skimage import io
from sklearn import svm
from sklearn.svm import NuSVC

def renameData(inFolder, outFolder):
    print('---RENAME FILE---')
    listDirs = os.listdir(inFolder)
    for subDir in listDirs:
        number = 1
        path = inFolder + subDir + '/'
        for imgfile in os.listdir(path):
            img = cv2.imread(path+imgfile)
            name,ext = os.path.splitext(imgfile)
            print(path + imgfile)
            if not os.path.exists(outFolder + subDir):
                os.makedirs(outFolder + subDir)
            try:
                cv2.imwrite(outFolder + subDir + '/' + str(number) + ext, img)
                number+=1
            except Exception as ex:
                print('Cannot save: ' + path + imgfile)
            
    print('---RENAME COMPLETE---')

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
                # Y_label = subDir #This line for show Y_train by name
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
    print('---SAVE VECTOR COMPLETE!---')

def testASample(imgfile):
    faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img = cv2.imread(imgfile)
    # Detect Faces
    faces = faceDetect.detectMultiScale(img)
    if (isinstance(faces, tuple)):
        print('CANNOT DETECT FACE: ' + imgfile)
    else:
        x, y, w, h = faces[0][:4]
        faceCrop = img[y:y + h, x:x + w]
        croppedImg = cv2.resize(faceCrop, (32, 32))
        cv2.imwrite('face'+imgfile, croppedImg)
        imgFace = io.imread('face'+imgfile, as_grey=True)
        X_test = imgFace.reshape(1,32*32)
        
        X_train, Y_train = faceToVetor('face/trainFace/')
        clf = NuSVC()
        print(clf.fit(X_train, Y_train))
        Y_pre = clf.predict(X_test)
        print('Y predict:')
        print(Y_pre)




def train(trainPath, testPath):
    X_train, Y_train = faceToVetor(trainPath)

    # faceDetectAndResizeImg(testPath, 'face/testFace/')
    # X_test, Y_test = faceToVetor('face/testFace/')
    X_test, Y_test = faceToVetor(testPath)
    # clf = svm.SVC()
    clf = NuSVC()
    print(clf.fit(X_train, Y_train))
    
    print('---TRAIN COMPLETE!---\n')
    Y_pre = clf.predict(X_test)
    print(clf.support_vectors_)

    print('Y Test:')
    print(Y_test)
    print('Y predict:')
    print(Y_pre)
    numTest = len(Y_test)
    testTrue = 0
    for x in range(0, numTest):
        if (Y_pre[x] == Y_test[x]):
            testTrue+=1
    print('Test True: ', testTrue, '/', numTest,'=', round(float(testTrue/numTest*100),3), '%')
    

renameData('rawdata/', 'renameData/')
# faceDetectAndResizeImg('train/', 'face/trainFace/')
# faceDetectAndResizeImg('validation/', 'face/valiFace/')
# faceDetectAndResizeImg('test/', 'face/testFace/')

# faceToVetor('face/trainFace/')

# writeVector('face/trainFace/', 'vectorTrain.txt')
# writeVector('face/valiFace/', 'vectorVali.txt')
# writeVector('face/testFace/', 'vectorTest.txt')

# train('/media/linhphan/LEARN/CNTT 2/IS YEAR 4 SEMESTER 1/ML- KhoatTQ/BTL/datasetFace/', 'face/testFace/')
# testASample('5.jpg')




