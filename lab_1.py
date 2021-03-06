import numpy as np
import matplotlib.pyplot as plt
import DataGenerator as dg


def generate():
    #Задаем исходные данные для линейного распределения
    mu_class_0 = [0, 2, 3]
    #mu_class_1 = [3, 5, 1]
    mu_class_1 = [4, 6, 0]
    mu = [mu_class_0, mu_class_1]
    sigma_class_0 = [2, 1, 2]
    sigma_class_1 = [1, 2, 1]
    sigma = [sigma_class_0, sigma_class_1]
    N = 1000
    col = len(mu_class_0)
    #Получаем массив линейно разделимых данных
    X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)

    #col = 2
    #X, Y, class0, class1 = dg.nonlinear_dataset_4(1, N)

    #Делим данные на test и train в пропорции 70/30
    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2 + 1]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:N * 2 + 1]
    #Построим гистограммы распределения классов по признакам
    for i in range(0, col):
        _ = plt.hist(class0[:, i], bins='auto', alpha=0.7)
        _ = plt.hist(class1[:, i], bins='auto', alpha=0.7)
        plt.title('Histogram feature ' + str(i+1))
        plt.xlabel('values')
        plt.ylabel('amount')
        plt.savefig('hist_f' + str(i + 1) + '.png')
        plt.clf()
    #Построим скаттерограммы распределения признаков для первых двух признаков
    plt.scatter(class0[:, 0], class0[:, 1], marker=".", alpha=0.7)
    plt.scatter(class1[:, 0], class1[:, 1], marker=".", alpha=0.7)
    plt.title('Scatter features 1st and 2nd')
    plt.xlabel('first')
    plt.ylabel('second')
    plt.savefig('scatter_CLASS_f_1_2.png')
    plt.clf()

    print('Finished')


if __name__ == '__main__':
    generate()