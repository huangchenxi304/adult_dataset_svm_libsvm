# adult_dataset_svm_libsvm

## Data preprocessing
* Basic way： only cleaning data (mostly deleting missing value), digitalization and discretization.
* Handling dummy variables besides process above

## SVM
* Using svm in python package sklearn (without adjustment parameters)
* Ten-fold cross validation

## libsvm
* Using [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/), a svm library developed by Professor Chih-Jen Lin of Taiwan in 2001
* The training set and test set are divided at a ratio of 2:1. 'a9a' is training set while 'a9a.t' is test set. This two sets were in the format of libsvm, which came from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a

## More
* adult_libsvm_self.py is a file that I try converting the data I preprocessed to libsvm format then using libsvm on it, but a little problems occured. Still improving...

## Thanks
appreciation for work on https://blog.csdn.net/gengkui9897/article/details/83049036 and https://blog.csdn.net/dianqijiaojianshuo/article/details/80620811
