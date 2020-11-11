import numpy as np
from sklearn.linear_model import LogisticRegression
import DataGenerator as dg
import matplotlib.pyplot as plt



def logit_liner():
    #Задаем исходные данные для линейного распределения
    mu_class_0 = [0, 2, 3]
    mu_class_1_bad = [3, 5, 1]
    mu_class_1_good = [4, 6, 0]
    mu = [mu_class_0, mu_class_1_bad]
    sigma_class_0 = [2, 1, 2]
    sigma_class_1 = [1, 2, 1]
    sigma = [sigma_class_0, sigma_class_1]
    N = 1000
    col = len(mu_class_0)
    #Получаем массив линейно разделимых данных
    X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)
    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2 + 1]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:N * 2 + 1]

    clf = LogisticRegression(random_state=4, solver='saga').fit(Xtrain, Ytrain)
    Pred_test = clf.predict(Xtest)
    Pred_train_proba = clf.predict_proba(Xtrain)
    Pred_test_proba = clf.predict_proba(Xtest)

    acc_train = clf.score(Xtrain, Ytrain)
    acc_test = clf.score(Xtest, Ytest)

    print(acc_train)
    print(acc_test)

    _ = plt.hist(Pred_train_proba[Ytrain,1], bins='auto', alpha=0.7)
    _ = plt.hist(Pred_train_proba[~Ytrain,1], bins='auto', alpha=0.7)
    plt.title('Histogram liner bad case TRAIN')
    plt.xlabel('probability')
    plt.ylabel('amount')
    plt.savefig('logit_liner_bad_train.png')
    plt.clf()

    _ = plt.hist(Pred_test_proba[Ytest,1], bins='auto', alpha=0.7)
    _ = plt.hist(Pred_test_proba[~Ytest,1], bins='auto', alpha=0.7)
    plt.title('Histogram liner bad case TEST')
    plt.xlabel('probability')
    plt.ylabel('amount')
    plt.savefig('logit_liner_bad_test.png')
    plt.clf()



def logit():
    N = 1000
    SIGMA = 1
    X, Y, class0, class1 = dg.nonlinear_dataset_4(SIGMA, N)

    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2 + 1]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:N * 2 + 1]

    clf = LogisticRegression(random_state=5, solver='saga').fit(Xtrain, Ytrain)
    Pred_test = clf.predict(Xtest)
    Pred_test_proba = clf.predict_proba(Xtest)

    acc_train = clf.score(Xtrain, Ytrain)
    acc_test = clf.score(Xtest, Ytest)

    _ = plt.hist(Pred_test_proba[Ytest,1], bins='auto', alpha=0.7)
    _ = plt.hist(Pred_test_proba[~Ytest,1], bins='auto', alpha=0.7)
    plt.savefig('logit.png')
    plt.clf()

if __name__ == '__main__':
    logit_liner()
