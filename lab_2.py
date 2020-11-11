import numpy as np
from sklearn.linear_model import LogisticRegression
import DataGenerator as dg
import matplotlib.pyplot as plt

#
#Функция оценки результатов работы Logit
def evaluate(name, pYproba, pY, Y):
    #Определяем точность, чувствительность, специфичность
    TP = sum(pY & Y)
    TN = sum(~pY & ~Y)
    FP = sum(pY & ~Y)
    FN = sum(~pY & Y)

    ACC = (TP + TN) / (TP + TN + FP + FN)
    SENS = TP / (TP + FN)
    SPEC = TN / (TN + FP)

    print(name)
    print("Object amount: " + str(len(Y)))
    print("ACC: " + str(ACC))
    print("SENS: " + str(SENS))
    print("SPEC: " + str(SPEC))

    #Строим гистограмму распределения вероятности по классам
    _ = plt.hist(pYproba[Y, 1], bins='auto', alpha=0.7)
    _ = plt.hist(pYproba[~Y, 1], bins='auto', alpha=0.7)
    plt.title('Histogram ' + name + ' case')
    plt.xlabel('probability')
    plt.ylabel('amount')
    plt.savefig('logit_' + name + '.png')
    plt.clf()

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
    Xtest = X[trainCount:N * 2]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:N * 2]

    clf = LogisticRegression(random_state=4, solver='saga').fit(Xtrain, Ytrain)

    Pred = clf.predict(Xtest)
    Pred_proba = clf.predict_proba(Xtest)
    evaluate('liner_bad_test', Pred_proba, Pred, Ytest)





def logit_nonliner():
    N = 1000
    SIGMA = 1
    X, Y, class0, class1 = dg.nonlinear_dataset_4(SIGMA, N)

    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:N * 2]

    clf = LogisticRegression(random_state=4, solver='saga').fit(Xtrain, Ytrain)

    Pred = clf.predict(Xtest)
    Pred_proba = clf.predict_proba(Xtest)
    evaluate('liner_nonlinear_test', Pred_proba, Pred, Ytest)

if __name__ == '__main__':
    logit_nonliner()
