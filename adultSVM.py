from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


# 数据数字化
def dataDigitize():
    # 获得原始数据
    adult_raw = pd.read_csv("adult_data.csv", header=None)

    # 为特征添加标题
    adult_raw.rename(columns={0: 'age', 1: 'workclass', 2: 'fnlwgt', 3: 'education', 4: 'education_number',
                              5: 'marriage', 6: 'occupation', 7: 'relationship', 8: 'race', 9: 'sex',
                              10: 'capital_gain', 11: 'apital_loss', 12: 'hours_per_week', 13: 'native_country',
                              14: 'income'}, inplace=True)

    # 清理数据，删除缺失值
    adult_cleaned = adult_raw.dropna()

    # 属性数字化
    adult_digitization = pd.DataFrame()

    target_columns = ['workclass', 'education', 'marriage', 'occupation', 'relationship', 'race', 'sex',
                      'native_country',
                      'income']
    for column in adult_cleaned.columns:
        if column in target_columns:
            unique_value = list(enumerate(np.unique(adult_cleaned[column])))
            dict_data = {key: value for value, key in unique_value}
            adult_digitization[column] = adult_cleaned[column].map(dict_data)
        else:
            adult_digitization[column] = adult_cleaned[column]

    # 去掉首行
    adult_digitization.drop(labels=0, inplace=True)

    # 重建索引
    adult_digitization = adult_digitization.reset_index(drop=True)

    return adult_digitization


# 将数据转换为libsvm格式(跑一次要5min......)
def toLibsvmFormat():
    adult_digitization = dataDigitize()

    print(adult_digitization)

    num = 1
    for column in adult_digitization.columns:
        row = 0

        if column == 'income':
            continue

        for item in adult_digitization[column]:
            adult_digitization.loc[row, column] = str(num) + ':' + str(item)

            row = row + 1

        print(num)
        num = num + 1

    print(adult_digitization)

    # 将收入label置于第一列
    temp = adult_digitization['income']
    adult_digitization['income'] = adult_digitization['age']
    adult_digitization['age'] = temp

    # 以csv形式存储
    adult_digitization.to_csv('D:\\学习\\机器学习\\adult_digitization.csv', header=None, index=None)


# 按n-1:1划分训练集和测试集
def dataPartition(n):
    # 读取处理好的libsvm格式的数据
    adult_libsvm = pd.read_csv('adult_digitization.csv', header=None)

    # 计算截断点
    slice_point = adult_libsvm.shape[0] // n * (n - 1)

    adult_train = adult_libsvm.loc[:slice_point - 1]
    adult_test = adult_libsvm.loc[slice_point:]

    adult_train.to_csv('D:\\学习\\机器学习\\adult_train.csv', header=None, index=None)

    adult_test.to_csv('D:\\学习\\机器学习\\adult_test.csv', header=None, index=None)


def main(adult_clf, name):
    adult_digitization = dataDigitize()

    print(adult_digitization)

    # 构造输入和输出
    X = adult_digitization[
        ['age', 'workclass', 'fnlwgt', 'education', 'education_number', 'marriage', 'occupation', 'relationship',
         'race',
         'sex', 'capital_gain', 'apital_loss', 'hours_per_week', 'native_country']]
    Y = adult_digitization[['income']]

    # 交叉验证
    preaccsvm = []
    num = 1
    kf = KFold(n_splits=10)

    for train, test in kf.split(X):
        X_train, X_test = X.loc[train], X.loc[test]
        Y_train, Y_test = Y.loc[train], Y.loc[test]

        adult_clf.fit(X_train, Y_train.values.ravel())

        # test_score = clf.score(X_test, Y_test)
        # print("test_score:" + str(test_score))
        test_predictions = adult_clf.predict(X_test)
        accuracy = accuracy_score(Y_test.values.ravel(), test_predictions)
        preaccsvm.append(accuracy)
        print(name + str(num) + "测试集准确率:  %s " % accuracy)
        num = num + 1

    print(name + "十折交叉平均准确率:  %s " % np.mean(np.array(preaccsvm)))


svmclf = svm.SVC(kernel='rbf', C=1)
# main(svmclf,'svm')

MNBclf = MultinomialNB()
# main(MNBclf,'MNB')

GNBclf = GaussianNB()
# main(GNBclf,'GNB')

BNBclf = BernoulliNB()
# main(BNBclf,'BNB')

dataPartition(3)
