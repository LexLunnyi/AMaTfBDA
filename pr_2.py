import pandas as pd
import time

def pr_2():

  table = pd.read_csv('/media/sf_shared/LETI/diabetes_data_upload.csv')
  print(table["Age"].mean())
  avg_male = table[table['Gender'] == 'Male']['Age'].mean()
  avg_female = table[table['Gender'] == 'Female']['Age'].mean()
  print(avg_male)
  print(avg_female)


if __name__ == '__main__':
  start = time.time()
  pr_2()
  end = time.time()
  print(end - start)