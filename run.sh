# python3 libsvm/grid.py -log2c -10,10,2 -log2g -10,10,2 -svmtrain 'libsvm/svm-train' vector/vectorvalid > log/log1
# python3 libsvm/grid.py -log2c 2,6,1 -log2g -10,-6,1 -svmtrain 'libsvm/svm-train' vector/vectorbysubset/vectorvalid > log/logsubset
# libsvm/svm-train -s 0 -c 16 -g 0.0078125 vector/vectortrain model >log/logtrain
libsvm/svm-predict vector/vectorOutFrame model testout
# mkdir outFrame/img
# ffmpeg -i hello.mp4 -vf fps=1 outFrame/img/%d.jpg
