import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import DataGenerator as dg
import matplotlib.pyplot as plt
import scikitplot as skplt

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
    #Выводим результат
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
    plt.savefig(name + '.png')
    plt.clf()

#
#Функция выполняет предсказание с помощью модели и выводит результаты
def showres(name, model, Xtrain, Xtest, Ytrain, Ytest):
    #Делаем предсказание на трейне
    Pred = model.predict(Xtrain)
    Pred_proba = model.predict_proba(Xtrain)
    evaluate('lab3_' + name + '_train', Pred_proba, Pred, Ytrain)
    #Делаем предсказание на тесте
    Pred = model.predict(Xtest)
    Pred_proba = model.predict_proba(Xtest)
    evaluate('lab3_' + name + '_test', Pred_proba, Pred, Ytest)
    #Строим ROC-кривую
    skplt.metrics.plot_roc_curve(Ytest, Pred_proba, figsize=(10, 10))
    plt.title('ROC ' + name)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig(name + '.png')
    plt.clf()
    #Определяем площадь под ROC-кривой
    AUC = roc_auc_score(Ytest, Pred_proba[:, 1])
    print('AUC ' + name + ':' + str(AUC))


#
#Работа с линейно-разделимыми данными
def liner():
    #Задаем исходные данные для линейного распределения
    mu = [[0, 2, 3], [3, 5, 1]]
    sigma = [[2, 1, 2], [1, 2, 1]]
    N = 1000
    X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)
    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:N * 2]
    #Выполняем классификацию с DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(Xtrain, Ytrain)
    showres('liner_tree', model, Xtrain, Xtest, Ytrain, Ytest)
    #Выполняем классификацию с RandomForestClassifier
    model = RandomForestClassifier(criterion="entropy", random_state=0, n_estimators=100).fit(Xtrain, Ytrain)
    showres('liner_forest', model, Xtrain, Xtest, Ytrain, Ytest)


#
#Работа с нелинейно-разделимыми данными
def nonliner():
    #Генерируем исходные данные
    N = 1000
    SIGMA = 1
    X, Y, class0, class1 = dg.nonlinear_dataset_4(SIGMA, N)
    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:N * 2]
    #Выполняем классификацию с DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(Xtrain, Ytrain)
    showres('nonliner_tree', model, Xtrain, Xtest, Ytrain, Ytest)
    #Выполняем классификацию с RandomForestClassifier
    model = RandomForestClassifier(criterion="entropy", random_state=0).fit(Xtrain, Ytrain)
    showres('nonliner_forest', model, Xtrain, Xtest, Ytrain, Ytest)


#
#Функция поиска количества деревьев, дающих максимальную точность
def find_best_n_estimators():
    #Задаем исходные данные для линейного распределения
    mu = [[0, 2, 3], [3, 5, 1]]
    sigma = [[2, 1, 2], [1, 2, 1]]
    N = 1000
    X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)
    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:N * 2]
    #В цикле меняем количество деревьев и определяем максимальную точность
    maxACC = 0;
    resN = 0;
    for n in range(1, 300, 10):
        #Выполняем классификацию с RandomForestClassifier
        model = RandomForestClassifier(criterion="entropy", random_state=0, n_estimators=n).fit(Xtrain, Ytrain)
        ACC = model.score(Xtest, Ytest)
        if (ACC > maxACC):
            maxACC = ACC
            resN = n;
    #Выводим результат
    print("N: " + str(resN) + " ACC: " + str(maxACC))



if __name__ == '__main__':
    liner()
    nonliner()
    find_best_n_estimators()
