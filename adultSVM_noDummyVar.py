# 进行哑变量处理和数字化处理

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB


def main(adult_clf,name):
    # 获得原始数据
    adult_raw = pd.read_csv("adult_data.csv", header=None)

    # 为特征添加标题
    adult_raw.rename(columns={0: 'age', 1: 'workclass', 2: 'fnlwgt', 3: 'education', 4: 'education_number',
                              5: 'marriage', 6: 'occupation', 7: 'relationship', 8: 'race', 9: 'sex',
                              10: 'capital_gain', 11: 'apital_loss', 12: 'hours_per_week', 13: 'native_country',
                              14: 'income'}, inplace=True)

    # 清理数据，删除缺失值
    adult_cleaned = adult_raw.dropna()

    # 去掉首行
    adult_cleaned.drop(labels=0, inplace=True)

    # 重建索引
    adult_cleaned = adult_cleaned.reset_index(drop=True)

    # print(adult_cleaned)

    # 处理哑变量
    target_columns = ['workclass', 'education', 'marriage', 'occupation', 'relationship', 'race', 'sex',
                      'native_country']
    for column in adult_cleaned.columns:
        if column in target_columns:
            adult_cleaned = pd.concat([adult_cleaned, pd.get_dummies(adult_cleaned[column])], axis=1)
            adult_cleaned.drop(labels=column, axis=1, inplace=True)

    # 将收入income转换为二值的label
    unique_value = list(enumerate(np.unique(adult_cleaned['income'])))
    dict_data = {key: value for value, key in unique_value}
    adult_cleaned['income'] = adult_cleaned['income'].map(dict_data)

    # 构造输入和输出
    Y = adult_cleaned[['income']]
    X = adult_cleaned.drop(labels='income', axis=1)

    # 交叉验证
    preaccsvm = []
    num = 1
    kf = KFold(n_splits=10)

    for train, test in kf.split(X):
        X_train, X_test = X.loc[train], X.loc[test]
        Y_train, Y_test = Y.loc[train], Y.loc[test]

        adult_clf.fit(X_train, Y_train.values.ravel())
        test_predictions = adult_clf.predict(X_test)
        accuracy = accuracy_score(Y_test.values.ravel(), test_predictions)
        preaccsvm.append(accuracy)
        print(name+"(处理哑变量)" + str(num) + "测试集准确率:  %s " % accuracy)
        num = num + 1

    print(name+"(处理哑变量)十折交叉平均准确率:  %s " % np.mean(np.array(preaccsvm)))

svmclf = svm.SVC(kernel='rbf', C=1)
#main(svmclf,'svm')

MNBclf = MultinomialNB(alpha=0.0001)
#main(MNBclf,'MNB')

GNBclf = GaussianNB()
main(GNBclf,'GNB')

BNBclf = BernoulliNB()
# main(BNBclf,'BNB')
