import csv
from tabulate import tabulate
import time

def pr_1():
  COLUMN_AGE = 0
  COLUMN_GENDER = 1
  COLUMN_OBESITY = 15
  COLUMN_DIABET = 16

  with open('/media/sf_shared/LETI/diabetes_data_upload.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
    print(len(data))

    age_male = 0
    age_female = 0
    cnt_male = 0
    cnt_female = 0
    res = [[0, 0], [0, 0]]

    for i in range(1, len(data)):
        res_obesity = 1
        res_diabet = 0
        gender = data[i][COLUMN_GENDER]
        if gender == "Male":
            age_male += int(data[i][COLUMN_AGE])
            cnt_male += 1
        if gender == "Female":
            age_female += int(data[i][COLUMN_AGE])
            cnt_female += 1
        if data[i][COLUMN_OBESITY] == "Yes":
            res_obesity = 0
        if data[i][COLUMN_DIABET] == "Positive":
            res_diabet = 1
        res[res_obesity][res_diabet] += 1

    avg_male = age_male / cnt_male
    avg_female = age_female / cnt_female
    print("Male -> " + str(avg_male))
    print("Female -> " + str(avg_female))

    print(tabulate([['Obesity Positive', res[0][0], res[0][1]], ['Obesity Negative', res[1][0], res[1][1]]], headers=['', 'Diabet Positive', 'Diabet Negative']))


if __name__ == '__main__':
    start = time.time()
    pr_1()
    end = time.time()
    print(end - start)