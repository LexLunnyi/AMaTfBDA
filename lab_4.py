import numpy as np
import DataGenerator as dg
import NeuralNetwork as nn
import matplotlib.pyplot as plt

def graphs(data, name):
    _ = plt.plot(data[:, 0], data[:, 1], label='LOSS')
    _ = plt.plot(data[:, 0], data[:, 2], label='ACC')
    plt.title(name + ' investigation')
    plt.xlabel(name)
    plt.ylabel('LOSS/ACC')
    plt.savefig(name + '_investigation.png')
    plt.clf()

def reshape(value):
    return np.reshape(value, [len(value), 1])

# Работа с линейно-разделимыми данными (поиск кол-ва эпох)
def liner_find_epoch():
    # Задаем исходные данные для линейного распределения
    mu = [[0, 2, 3], [4, 6, 0]]
    sigma = [[2, 1, 2], [1, 2, 1]]
    N = 1000
    X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)
    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2]
    Ytrain = reshape(Y[0:trainCount])
    Ytest = reshape(Y[trainCount:N * 2])

    # инициализируем ИНС
    model = nn.NeuralNetwork(Xtrain, Ytrain, 4)
    N_epoch = 150
    # тренируем сети и получаем точность
    data = np.empty((N_epoch, 3))
    for i in range(N_epoch):
        # итерация обучения
        model.train(0.005)
        # получаем точность
        LOSS, ACC = model.test(Xtest, Ytest)
        data[i][0] = i
        data[i][1] = LOSS
        data[i][2] = ACC
    graphs(data, 'epoch')


# Работа с линейно-разделимыми данными (поиск кол-ва нейронов)
def liner_find_n_cnt():
    # Задаем исходные данные для линейного распределения
    mu = [[0, 2, 3], [4, 6, 0]]
    sigma = [[2, 1, 2], [1, 2, 1]]
    N = 1000
    X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)
    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2]
    Ytrain = reshape(Y[0:trainCount])
    Ytest = reshape(Y[trainCount:N * 2])
    # задаем максимальное кол-во нейронов
    N_max = 20
    # тренируем сети и получаем точность
    data = np.empty((N_max, 3))
    for i in range(N_max):
        # инициализируем ИНС
        model = nn.NeuralNetwork(Xtrain, Ytrain, i)
        # итерация обучения
        model.full_train(20, 0.005)
        # получаем точность
        LOSS, ACC = model.test(Xtest, Ytest)
        data[i][0] = i
        data[i][1] = LOSS
        data[i][2] = ACC
    graphs(data, 'neural')

# Работа с линейно-разделимыми данными (вывод точности и весов)
def liner_find_final():
    # Задаем исходные данные для линейного распределения
    mu = [[0, 2, 3], [4, 6, 0]]
    sigma = [[2, 1, 2], [1, 2, 1]]
    N = 1000
    X, Y, class0, class1 = dg.norm_dataset(mu, sigma, N)
    trainCount = round(0.7 * N * 2)
    Xtrain = X[0:trainCount]
    Xtest = X[trainCount:N * 2]
    Ytrain = reshape(Y[0:trainCount])
    Ytest = reshape(Y[trainCount:N * 2])
    # инициализируем ИНС
    model = nn.NeuralNetwork(Xtrain, Ytrain, 5)
    # итерация обучения
    model.full_train(20, 0.005)
    # получаем точность
    LOSS, ACC = model.test(Xtest, Ytest)
    print('RESULTS:\n')
    print('LOSS -> ' + str(LOSS) + '\n')
    print('ACC -> ' + str(ACC) + '\n')
    # получаем веса
    model.print()

if __name__ == '__main__':
    liner_find_epoch()
    liner_find_n_cnt()
    liner_find_final()
