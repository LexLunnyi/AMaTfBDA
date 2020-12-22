import numpy as np

class NeuralNetwork:
    def __init__(self, x, y, n_cnt):
        self.x = x
        self.y = y
        # инициализируем веса первого слоя
        self.W1 = np.random.rand(x.shape[1], n_cnt)
        # инициализируем веса второго слоя
        self.W2 = np.random.rand(n_cnt, 1)
        # инициализируем массив для предсказанных значений
        self.output = np.zeros(y.shape)

    # Активационная функция сигмоиды
    def __S(self, z):
        return 1/(1+np.exp(-z))

    # Функция - производная сигмоиды
    def __derS(self, p):
        return p * (1 - p)

    #получение предсказания
    def __feedforward(self, curX):
        # Расчитываем значение на выходе первого слоя
        self.L1 = self.__S(np.dot(curX, self.W1))
        # Расчитываем значение на выходе второго слоя
        self.L2 = self.__S(np.dot(self.L1, self.W2))
        return self.L2

    # обучение обратным распределением
    def __backprop(self, out, koeff):
        # рассчитываем обратный градиент для весов второго слоя
        dW2 = np.dot(self.L1.T, koeff*(self.y - out)*self.__derS(out))
        # рассчитываем обратный градиент для весов первого слоя
        dW1 = np.dot(self.x.T, np.dot(koeff*(self.y - out)*self.__derS(out), self.W2.T)*self.__derS(self.L1))
        # обновляем веса
        self.W1 += dW1
        self.W2 += dW2

    # тренировка ИНС
    def train(self, koeff):
        # рассчитываем значения на выходе сети и тренируем ее обратным распространением
        self.__backprop(self.__feedforward(self.x), koeff)

    # тренировка ИНС с заданным кол-вом эпох
    def full_train(self, n_epoch, koeff):
        for i in range(n_epoch):
            self.train(koeff)

    # тестирование ИНС
    def test(self, Xtest, Ytest):
        # получаем предсказание
        out = self.__feedforward(Xtest)
        pred = np.rint(out)
        # определяем потери
        LOSS = np.mean(np.square(Ytest - out))
        # определяем точность
        ACC = sum(pred == Ytest) / len(Ytest)
        return LOSS, ACC

    # выводим веса
    def print(self):
        print('W1 -> ' + str(self.W1) + '\n')
        print('W2 -> ' + str(self.W2) + '\n')