# Identify the singer in the video SVM
-------------------------------------
PROJECT NAME:
Identify the singer in the video SVM

Enviroment: Ubuntu 16.04
Library: Opencv 3.3.0, Libsvm 3.21, Skimage
Language: Python 3.6
Tool: FFMPEG : sudo apt-get install ffmpeg
-------------------------------------
PREPROCESS: preProcess.py
- rawdata/ : raw data folder . Each folder have a person
First using renamedata('rawdata/', 'dataset/') in preProcess.py to rename data into 1.jpg, 2.jpg
==> We have dataset folder.

Then run faceDetectAndResizeImg('dataset/', 'face/') to dectect and resize face into 32x32 px.
Folder 'face/' now have all face size 32x32 px.

Next, we want to write all face into vectorFull.
Run: writeVector('face/', 'vectorFull')
Now, we have vectorFull in folder 'vector/'.

END PREPROCESS
-------------------------------------
TRAINING MODEL: TrainModel.sh

Run: ./libsvm/subset.py vector/vectorFull 200 vector/vectorTest vector/vectorTrain
to divide vectorFull to vectorTest and vectorTrain.

Run: python3 libsvm/grid.py -log2c 4,8,1 -log2g -12,-8,1 -v 5 -svmtrain 'libsvm/svm-train' vector/vectorTrain > log/logParameter3
we have C and gamma parameter in log/logPrameter3:
C=32.0, g = 0.00390625 with Accurancy = 82.3358%

Run: libsvm/svm-train -s 0 -c 32 -g 0.00390625 vector/vectorTrain model
to trainning data.
Output: file model

TEST with vectorTest:
Run: libsvm/svm-predict vector/vectorTest model testout
Output: file testout contain all label.

END TRAINING MODEL 
-------------------------------------
PREDICT A VIDEO : predictVideo.py

if you want to predict a video, just run: python3 predictVideo.py
Note that you've change parameter in function predict('file video name')
Ex: change function to: predict('hello.mp4')
Then run: python3 predictVideo.py
OUTPUT: LABEL: Adele!
-------------------------------------
Contact me: fb.com/deluxepsk
Email: linhphanhust@gmail.com
