# Divide data to train, test, validation
./libsvm/subset.py vector/vectorFull 200 vector/vectorTest vector/vectorTrain

# parameter Optimization
# python3 libsvm/grid.py -log2c -50,50,10 -log2g -50,50,10 -v 5 -svmtrain 'libsvm/svm-train' vector/vectorTrain > log/logParameter1
# python3 libsvm/grid.py -log2c 0,20,2 -log2g -20,0,2 -v 5 -svmtrain 'libsvm/svm-train' vector/vectorTrain > log/logParameter2
# python3 libsvm/grid.py -log2c 4,8,1 -log2g -12,-8,1 -v 5 -svmtrain 'libsvm/svm-train' vector/vectorTrain > log/logParameter3



# Train and test model:
libsvm/svm-train -s 0 -c 32 -g 0.00390625 vector/vectorTrain model >log/trainResult2
libsvm/svm-predict vector/vectorTest model testout
