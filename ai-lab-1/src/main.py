from art import tprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


TITLE = 'Lab 1'
SOURCE_FILE = '../08_Сегментация покупателей/customers.xls'
OBJECTS_COUNT = 184
COLUMN_NAMES = ('пол код', 'возраст код', 'Телеканал', 'Профессия.1', 'Пресса')
OBJECTS_DISTANCE_TYPE = 'euclidean'
CLUSTERS_DISTANCE_TYPE = 'ward'


def main():
  # data import
  df = pd.read_excel(SOURCE_FILE, nrows=OBJECTS_COUNT, usecols=COLUMN_NAMES)

  # hierarcical clustering
  scaler = preprocessing.MinMaxScaler().fit(df.to_numpy())
  scaled_data = scaler.transform(df.to_numpy())
  distance_matrix = linkage(scaled_data, method=CLUSTERS_DISTANCE_TYPE, metric=OBJECTS_DISTANCE_TYPE)

  # dendrogram plotting
  fig = plt.figure(figsize=(15, 30))
  fig.patch.set_facecolor('white')
  dendrogram(distance_matrix,
             orientation='top',
             leaf_font_size=12,)

  # k-means clustering
  

  plt.show()


if __name__ == '__main__':
  tprint(TITLE)
  main()
