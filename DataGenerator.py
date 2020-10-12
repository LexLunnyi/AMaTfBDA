import numpy as np
import math

#функция генерации линейно разделимых массивов данных
def norm_dataset(mu, sigma, N):
    mu_class_0 = mu[0]
    mu_class_1 = mu[1]
    sigma_class_0 = sigma[0]
    sigma_class_1 = sigma[1]
    #Определяем число столбцов-признаков
    col = len(mu_class_0)
    #Генерируем данные для классов с помощью нормально распределения
    class0 = np.random.normal(mu_class_0[0], sigma_class_0[0], [N, 1])
    class1 = np.random.normal(mu_class_1[0], sigma_class_1[0], [N, 1])
    for i in range(1, col):
        v0 = np.random.normal(mu_class_0[i], sigma_class_0[i], [N, 1])
        class0 = np.hstack((class0, v0)) #Добавляем новый столбец-признак
        v1 = np.random.normal(mu_class_1[i], sigma_class_1[i], [N, 1])
        class1 = np.hstack((class1, v1)) #Добавляем новый столбец-признак
    #Генерируем метки для кдассов
    Y1 = np.ones((N, 1), dtype=bool)
    Y0 = np.zeros((N, 1), dtype=bool)
    #Сводим в единые переменные объекты двух классов и метки
    X = np.vstack((class0, class1))
    Y = np.vstack((Y0, Y1)).ravel()
    #Перемешиваем массивы данных
    rng = np.random.default_rng()
    arr = np.arange(2 * N)
    rng.shuffle(arr)
    X = X[arr]
    Y = Y[arr]
    #Возвращаем данные
    return X, Y, class0, class1



# Функция генерирует массивы матожиданий для двух классов
# Распределение матожидание по оси первого признака
#
#                  |---------|   class1
#                      len1      len1 = len0 / 2
#  |<------------->|             noffset (normalized offset, as part of len0)
#  |------------------|          class0
#         len0
#
def get_mus(len0, offset, N):
    len1 = len0 / 2
    X0_BEGIN = 4
    X1_BEGIN = offset * len0 + X0_BEGIN
    x0 = X0_BEGIN
    y0 = x0 / 2 + 2
    x1 = X1_BEGIN
    y1 = x1
    mu0 = [x0, y0]
    mu1 = [x1, y1]
    #Генрируем данные для "левой" половины графика
    for i in range(1, round(N/2)):
        x0 = X0_BEGIN + i * 2 * len0 / N
        y0 = x0 / 2 + 2
        x1 = X1_BEGIN + i * len1 / N
        y1 = x1
        mu0 = np.vstack((mu0, [x0, y0]))
        mu1 = np.vstack((mu1, [x1, y1]))
    #Генрируем данные для "правой" половины графика
    for i in range(round(N/2), N):
        x0 = X0_BEGIN + (i - N/2) * 2 * len0 / N / math.sqrt(2)
        y0 = x0 * 2 - 2
        x1 = X1_BEGIN + i * len1 / N
        y1 = x1
        mu0 = np.vstack((mu0, [x0, y0]))
        mu1 = np.vstack((mu1, [x1, y1]))
    return mu0, mu1





#функция генерации линейно неразделимых массивов данных
def nonlinear_dataset_4(sigma, N):
    CLASS_LEN = 10
    CLASS_OFFSET = 0.8
    #Определяем координаты матожиданий для получения необходимого распределения
    mu0, mu1 = get_mus(CLASS_LEN, CLASS_OFFSET, N)
    col = 2
    sigma01 = [sigma, sigma]
    #Генерируем данные для классов с помощью нормально распределения
    class0 = np.random.normal(mu0[0], sigma01, [1, col])
    class1 = np.random.normal(mu1[0], sigma01, [1, col])
    for i in range(1, N):
        v0 = np.random.normal(mu0[i], sigma01, [1, col])
        class0 = np.vstack((class0, v0))
        v1 = np.random.normal(mu1[i], sigma01, [1, col])
        class1 = np.vstack((class1, v1))
    #Генерируем метки для кдассов
    Y1 = np.ones((N, 1), dtype=bool)
    Y0 = np.zeros((N, 1), dtype=bool)
    #Сводим в единые переменные объекты двух классов и метки
    X = np.vstack((class0, class1))
    Y = np.vstack((Y0, Y1)).ravel()
    #Перемешиваем массивы данных
    rng = np.random.default_rng()
    arr = np.arange(2 * N)
    rng.shuffle(arr)
    X = X[arr]
    Y = Y[arr]
    return X, Y, class0, class1
