import numpy as np
import matplotlib.pyplot as plt
import DataGenerator as dg


def generate():
    mu0 = [0, 2, 3]
    mu1 = [3, 5, 1]
    mu = [mu0, mu1]
    sigma0 = [2, 1, 2]
    sigma1 = [1, 2, 1]
    sigma = [sigma0, sigma1]
    N = 1000
    #col = len(mu0)
    col = 2
    #X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)
    X, Y, class0, class1 = dg.nonlinear_dataset_4(1, N)

    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2 + 1]
    Ytrain = Y[0:trainCount]
    Ytest = Y[trainCount:N * 2 + 1]

    for i in range(0, col):
        _ = plt.hist(class0[:, i], bins='auto', alpha=0.7)
        _ = plt.hist(class1[:, i], bins='auto', alpha=0.7)
        plt.title('Histogram feature ' + str(i+1))
        plt.xlabel('values')
        plt.ylabel('count')
        plt.savefig('hist_f' + str(i + 1) + '.png')
        plt.clf()

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