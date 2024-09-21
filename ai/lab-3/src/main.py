from art import tprint
import pandas as pd
from enum import StrEnum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from dataclasses import dataclass
import statistics


TITLE = 'Lab 3'
SOURCE_FILE_NAME = '..\\11_2004 New_Car_and_Truck_Data_SUV\\04cars.dat'
CSV_SEPARATOR = ';'
class column_names(StrEnum):
  VEHICLE_NAME = 'Vehicle Name'
  SPORTS_CAR = 'Sports Car'
  SPORT_UTILITY_VEHICLE = 'Sport Utility Vehicle'
  WAGON = 'Wagon'
  MINIVAN = 'Minivan'
  PICKUP = 'Pickup'
  ALL_WHEEL_DRIVE = 'All-Wheel Drive'
  REAR_WHEEL_DRIVE = 'Rear-Wheel Drive'
  SUGGESTED_RETAIL_PRIC = 'Suggested Retail Pric'
  DEALER_COST = 'Dealer Cost'
  ENGINE_SIZE = 'Engine Siz'
  NUMBER_OF_CYLINDERS = 'Number of Cylinders'
  HORSEPOWER = 'Horsepower'
  CITY_MILES_PER_GALLON = 'City Miles Per Gallon'
  HIGHWAY_MILES_PER_GALLON = 'Highway Miles Per Gallon'
  WEIGHT = 'Weight'
  WHEEL_BASE = 'Wheel Base'
  LENGTH = 'Length'
  WIDTH = 'Width'

  def bool_columns():
    return (
      column_names.SPORTS_CAR,
      column_names.SPORT_UTILITY_VEHICLE,
      column_names.MINIVAN,
      column_names.PICKUP,
    )
  
  def train_columns():
    return (
      column_names.ALL_WHEEL_DRIVE,
      column_names.ENGINE_SIZE,
      column_names.NUMBER_OF_CYLINDERS,
      column_names.HORSEPOWER,
      column_names.WEIGHT,
      column_names.WHEEL_BASE,
    )


@dataclass
class Data:
  values: pd.DataFrame
  classes: pd.DataFrame
  total: int
  sport_utilities: int
  other: int


def main():
  df = get_data()
  total, su, other = describe(df)
  name = 'Исходный'
  print_describe(name, total, su, other)

  df_accuracies, target_k = find_target_k(df)

  print('Тест точности при разных значениях k (по 50 запусков)')
  print(df_accuracies)
  plt.plot(df_accuracies['k'], df_accuracies['accuracy'])
  print(f'Max k is {target_k}')

  train, real = get_train_and_real_data(df)
  for name, d in (('Обучающий', train), ('Тренировочный', real)):
    print_describe(name, d.total, d.sport_utilities, d.other) 

  # contingency_table = pd.crosstab(df[column_names.ALL_WHEEL_DRIVE], df[column_names.WHEEL_BASE])
  # print(contingency_table)
  
  train_accuracy, real_accuracy = classify(target_k, train, real)

  print(f'Точность на обучающих данных: {train_accuracy}')
  print(f'Точноcть на тренировочных данных: {real_accuracy}')
  plt.show()

def classify(target_k, train, real):
    df_train, train_classes = train.values, train.classes
    df_real, real_classes = real.values, real.classes
    knn = KNeighborsClassifier(n_neighbors=target_k)
    knn_model = knn.fit(df_train, train_classes)

    knn_accuracy_test_predictions = knn.predict(df_train)
    train_accuracy = accuracy_score(train_classes, knn_accuracy_test_predictions)

    knn_prediction = knn.predict(df_real)
    real_accuracy = accuracy_score(real_classes, knn_prediction)
    return train_accuracy,real_accuracy

def find_target_k(df):
    df_accuracies = pd.DataFrame(columns=('k', 'accuracy'))
    accuracies = []
    for k in range(5, 11):
      temp = []
      for _ in range(50):
        train, real = get_train_and_real_data(df)
        df_train, train_classes = train.values, train.classes
        df_real, real_classes = real.values, real.classes
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(df_train, train_classes)
        knn_accuracy_test_predictions = knn.predict(df_train)
        temp.append(accuracy_score(train_classes, knn_accuracy_test_predictions))
      accuracies.append(statistics.mean(temp))

    train, real = get_train_and_real_data(df)
  
    df_accuracies['k'] = list(range(5, 11))
    df_accuracies['accuracy'] = accuracies

    idx = df_accuracies['accuracy'].idxmax()
    target_k = df_accuracies['k'][idx]
    return df_accuracies,target_k

def print_describe(name, total, su, other):
    print(f'Набор данных "{name}"')
    print(f'Всего {total}')
    print(f'Внедорожников {su} ({su / total * 100:.0f}%)')
    print(f'Остальных {other} ({other / total * 100:.0f}%)')
    print()


def get_prediction_name(v: bool):
  return 'SPORT_UTILITY' if v else 'OTHER'


def describe(df: pd.DataFrame):
  total = len(df)
  sport_utility = len(df[df[column_names.SPORT_UTILITY_VEHICLE] == True])
  other = total - sport_utility
  return total, sport_utility, other


def prepare_for_prediction(df: pd.DataFrame):
  df = df.filter(items=column_names.train_columns())
  df.fillna(0, inplace=True)
  return df


def get_train_and_real_data(df):
  df_train_sport_utility = df[df[column_names.SPORT_UTILITY_VEHICLE] == True].sample(6)
  df_train_other = df[df[column_names.SPORT_UTILITY_VEHICLE] == False].sample(36)
  df_train = pd.concat([df_train_sport_utility, df_train_other])
  total_train, su_train, other_train = describe(df_train)
  train_classes = df_train[column_names.SPORT_UTILITY_VEHICLE].apply(get_prediction_name)
  temp = df_train.copy()
  df_train = prepare_for_prediction(df_train)
  data_train = Data(df_train, train_classes, total_train, su_train, other_train)

  df_real = pd.concat([df, temp]).drop_duplicates(keep=False)
  total_real, su_real, other_real = describe(df_real)
  real_classes = df_real[column_names.SPORT_UTILITY_VEHICLE].apply(get_prediction_name)
  df_real = prepare_for_prediction(df_real)
  data_real = Data(df_real, real_classes, total_real, su_real, other_real)

  return data_train, data_real


def get_data():
  df = pd.read_csv(SOURCE_FILE_NAME, sep=CSV_SEPARATOR, names=[v for v in column_names])
  df[column_names.VEHICLE_NAME] = df[column_names.VEHICLE_NAME].apply(lambda v: v.strip())
  for c in column_names.bool_columns():
    df[c] = df[c].astype(bool)
  return df

if __name__ == '__main__':
  tprint(TITLE)
  main()
