import numpy as np
from sklearn.linear_model import LogisticRegression
import DataGenerator as dg
import matplotlib.pyplot as plt


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
    logit()
