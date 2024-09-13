from art import tprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


SOURCE_FILE = '../08_Сегментация покупателей/customers.xls'
OBJECTS_COUNT = 184
COLUMN_NAMES = ('пол код', 'возраст код', 'Телеканал', 'Профессия.1', 'Пресса')


def main():
  df = pd.read_excel(SOURCE_FILE, nrows=OBJECTS_COUNT, usecols=COLUMN_NAMES)

  print('Imported data')
  print(df.head())

  scaler = preprocessing.MinMaxScaler().fit(df.to_numpy())
  scaled_data = scaler.transform(df.to_numpy())
  print('Transformed data')
  print(pd.DataFrame(scaled_data))

  distance_matrix = linkage(scaled_data, method='ward', metric='euclidean')

  fig = plt.figure(figsize=(15, 30))
  fig.patch.set_facecolor('white')
  R = dendrogram(distance_matrix,
                 orientation='top',
                 leaf_font_size=12,)
  plt.show()


if __name__ == '__main__':
  tprint('Lab 1')
  main()
