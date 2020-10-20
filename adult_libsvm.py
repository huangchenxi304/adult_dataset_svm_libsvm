
from libsvm.python.commonutil import svm_read_problem
from libsvm.python.svmutil import svm_train, svm_predict

y ,x = svm_read_problem('a9a')
m = svm_train(y ,x ,'-c 5')

test_y ,test_x = svm_read_problem('a9a.t')
p_label ,p_acc ,p_val = svm_predict(test_y ,test_x ,m)
