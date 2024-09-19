# Author: Pavel Putin <pavelputin2003@yandex.ru>

from art import tprint
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


TITLE = 'Lab 1'
SOURCE_FILE = '../08_Сегментация покупателей/customers.xls'
OBJECTS_COUNT = 184
SHEET_NAME = 'база спсс'
COLUMN_NAMES = ('пол код', 'возраст код', 'Телеканал', 'Пресса')
OBJECTS_DISTANCE_TYPE = 'euclidean'
CLUSTERS_DISTANCE_TYPE = 'ward'
# Определён на основе "локтя" и "силуэта"
K_MEANS_CLUSTERS = 3
K_MEANS_CLUSTER_LABELS = 'cluster_labels'

DECODE = {
  'пол код': {
    1: 'женский',
    0: 'мужской'
  }, 'возраст код': {
    1: 'младше 25',
    2: 'от 25 до 34',
    3: 'от 35 до 44',
    4: 'от 45 до 54',
    5: 'старше 54',
  }, 'Телеканал': {
    1: 'развлекательный (СТС, ТНТ, МТВ, МузТВ, спорт)',
    2: 'государственный (ОРТ, РТР)',
    3: 'частный (НТВ)',
    4: 'познавательный (дискавери, Культура)',
    5: 'все',
  }, 'Профессия': {
    1: 'высококвалифицированный труд',
    2: 'квалифицированный труд',
    3: 'неквалифицированный',
  }, 'Пресса': {
    1: 'деловая',
    2: 'глянцевые журналы',
    3: 'федеральные газеты (АиФ, КП)',
    4: 'профессиональные',
    5: 'афиша, программа',
  }
}


def main():
  # data import
  df = pd.read_excel(SOURCE_FILE, sheet_name=SHEET_NAME, nrows=OBJECTS_COUNT, usecols=COLUMN_NAMES)

  # hierarcical clustering
  linkage_matrix = make_hierarciacal_clustering(df)
  # dendrogram plotting
  plot_dendrogram(linkage_matrix)

  # k-means clustering
  scaled_data, WCSS, Silh, result = make_k_means_clustering(df)
  # k-means plotting
  plot_k_means(df, scaled_data, WCSS, Silh, result)

  describe_k_means_clusters(df)

  plt.show()


def describe_k_means_clusters(df):
    grouped = df.groupby(K_MEANS_CLUSTER_LABELS)
    clusters_info = []
    for category, group_df in grouped:
      cluster = {'Размер': len(group_df)}
      for column in COLUMN_NAMES:
        cluster[column] = int(round(group_df.describe()[column]['50%'], 0))
      clusters_info.append(cluster)

    for i, cluster in enumerate(clusters_info):
      print(f'Кластер {i + 1} - размер {cluster['Размер']} ({round(cluster['Размер'] / OBJECTS_COUNT * 100, 0)}%)')
      for col, value in cluster.items():
        if col != 'Размер':
          print(f'{col}: {DECODE[col][value]}')
      print()


def plot_k_means(df, scaled_data, WCSS, Silh, result):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(range(2, 11), WCSS)
    axes[0].set_title('Метод локтя')
    axes[0].set_ylabel('WCSS')
    axes[0].set_xlabel('Число кластеров')

    axes[1].plot(range(2, 11), Silh)
    axes[1].set_title('Индекс кластерного силуэта')
    axes[1].set_ylabel('Silhouette score')
    axes[1].set_xlabel('Число кластеров')

    # Многомерное шкалирование
    mds = MDS(random_state=1)
    pos = mds.fit_transform(scaled_data)
    centers = mds.fit_transform(result.cluster_centers_)

    axes[2].scatter(pos[:,0], pos[:,1],
              c=df[K_MEANS_CLUSTER_LABELS], cmap=plt.cm.Set1)
    axes[2].scatter(centers[:,0],
              centers[:,1],
              s=100, c='black', label='Centroids')
    for i, txt in enumerate(('0', '1', '2')):
      axes[2].annotate(txt, (centers[:,0][i], centers[:,1][i]), color='red')
    axes[2].set_title('Результат кластеризации (k-средних)')
    axes[2].set_xlabel('MDS ось x')
    axes[2].set_ylabel('MDS ось y')

    fig.savefig('k-means.png')


def make_k_means_clustering(df):
    scaler = preprocessing.MinMaxScaler().fit(df.to_numpy())
    scaled_data = scaler.transform(df.to_numpy())

    # Сумма квадратов внутрикластерных расстойний WCSS
    elbow = []
    silhouette = []
    for i in range(2, 11):
      kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
      kmeans.fit(scaled_data)
      elbow.append(kmeans.inertia_)
      silhouette.append(silhouette_score(scaled_data, 
                                         kmeans.fit_predict(scaled_data), 
                                         metric='euclidean'))

    result = KMeans(n_clusters=K_MEANS_CLUSTERS, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    result.fit(scaled_data)
    df[K_MEANS_CLUSTER_LABELS] = result.fit_predict(scaled_data)

    return scaled_data, elbow, silhouette, result


def plot_dendrogram(distance_matrix):
  fig = plt.figure(figsize=(15, 5))
  fig.patch.set_facecolor('white')
  dendrogram(distance_matrix,
            orientation='top',
            leaf_font_size=12,)
  fig.savefig('dendrogram.png')


def make_hierarciacal_clustering(df):
  scaler = preprocessing.MinMaxScaler().fit(df.to_numpy())
  scaled_data = scaler.transform(df.to_numpy())
  return linkage(scaled_data,
                 method=CLUSTERS_DISTANCE_TYPE,
                 metric=OBJECTS_DISTANCE_TYPE)


if __name__ == '__main__':
  tprint(TITLE)
  main()
