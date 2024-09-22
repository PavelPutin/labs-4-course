from cmath import isnan
from art import tprint
import pandas as pd
from enum import StrEnum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from dataclasses import dataclass
import statistics
import my_module__stat as stat_utils



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
  full_values: pd.DataFrame
  classes: pd.DataFrame
  total: int
  sport_utilities: int
  other: int


def engine_size_transform_func(engine_size):
  engine_size_dict = {
    'small': 1.7,
    'medium': 3.5,
    'large': 100
  }
  size_scale = list(engine_size_dict.values())
  if not engine_size or isnan(engine_size):
    result = None
  else:
    for i, elem in enumerate(size_scale):
      if abs(engine_size) <= elem:
        result = list(engine_size_dict.keys())[i]
        break
  return result


def wheel_base_transform_func(wheel_base_inch):
  wheel_base_dict_mm = {
    'A-class': 2380,
    'B-class': 2500,
    'C-class': 2700,
    'D-class': 2850,
    'E-class': 2950,
    'F-class': 10000
  }
  size_scale = list(wheel_base_dict_mm.values())
  if not wheel_base_inch or isnan(wheel_base_inch):
    result = None
  else:
    wheel_base_mm = wheel_base_inch * 25.4
    for i, elem in enumerate(size_scale):
      if abs(wheel_base_mm) <= elem:
        result = list(wheel_base_dict_mm.keys())[i]
        break
  return result


def horse_power_transform_func(horse_power):
  horse_power_dict = {
    'A-class': 100,
    'B-class': 150,
    'C-class': 200,
    'D-class': 250,
    'E-class': 10_000
  }
  size_scale = list(horse_power_dict.values())
  if not horse_power or isnan(horse_power):
    result = None
  else:
    for i, elem in enumerate(size_scale):
      if abs(horse_power) <= elem:
        result = list(horse_power_dict.keys())[i]
        break
  return result


def weight_transform_func(weight):
  weight_dict = {
    'mini': 1999,
    'light': 2499,
    'compact': 2999,
    'medium': 3499,
    'heavy (suv, pickup)': 100_000
  }
  size_scale = list(weight_dict.values())
  if not weight or isnan(weight):
    result = None
  else:
    for i, elem in enumerate(size_scale):
      if abs(weight) <= elem:
        result = list(weight_dict.keys())[i]
        break
  return result


def main():
  df = get_data()
  
  total, su, other = describe(df)
  name = 'Исходный'
  print_describe(name, total, su, other)

  df_accuracies, target_k = find_target_k(df)

  print('Тест точности при разных значениях k (по 50 запусков)')
  print(df_accuracies)
  plt.plot(df_accuracies['k'], df_accuracies['accuracy'])
  plt.savefig('accuracies.png')
  plt.show()

  train, real = get_train_and_real_data(df)
  contingency_analyze(real.full_values)
  for name, d in (('Обучающий', train), ('Тренировочный', real)):
    print_describe(name, d.total, d.sport_utilities, d.other) 

  # contingency_table = pd.crosstab(df[column_names.ALL_WHEEL_DRIVE], df[column_names.WHEEL_BASE])
  # print(contingency_table)
  
  train_accuracy, real_accuracy = classify(target_k, train, real)

  print(f'Точность на обучающих данных: {train_accuracy}')
  print(f'Точноcть на тренировочных данных: {real_accuracy}')

def contingency_analyze(df):
  working_df = df.copy()
  working_df['Engine size category'] = working_df[column_names.ENGINE_SIZE].apply(engine_size_transform_func)
  working_df['Wheel base classes'] = working_df[column_names.WHEEL_BASE].apply(wheel_base_transform_func)
  working_df['Horse power classes'] = working_df[column_names.HORSEPOWER].apply(horse_power_transform_func)
  working_df['Weight classes'] = working_df[column_names.WEIGHT].apply(weight_transform_func)

  temp = (
    ('Engine size category', 'Связь размера двигателя и типа автомобиля', 'engine_size_type.png'),
    (column_names.ALL_WHEEL_DRIVE, 'Связь полноприводности и типа автомобиля', 'all_wheel_type.png'),
    ('Wheel base classes', 'Связь колёсной базы и типа автомобиля', 'wheel_base_type.png'),
    (column_names.NUMBER_OF_CYLINDERS, 'Связь количества цилиндров и типа автомобиля', 'cylinders_type.png'),
    ('Horse power classes', 'Связь лошадиных сил и типа автомобиля', 'horsepower_type.png'),
    ('Weight classes', 'Связь веса и типа автомобиля', 'weight_type.png'),
  )

  for columns, title_figure, file_name in temp:
    temp_df = working_df.pivot_table(
      values=column_names.VEHICLE_NAME,
      index=column_names.SPORT_UTILITY_VEHICLE,
      columns=columns,
      aggfunc='count',
      fill_value=0,
      margins=True
    )

    stat_utils.graph_contingency_tables_heatmap(
      data_df_in=temp_df,
      title_figure=title_figure,
      file_name=file_name
    )


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
  df_train_sport_utility = df[df[column_names.SPORT_UTILITY_VEHICLE] == True].sample(12)
  df_train_other = df[df[column_names.SPORT_UTILITY_VEHICLE] == False].sample(72)
  df_train = pd.concat([df_train_sport_utility, df_train_other])
  total_train, su_train, other_train = describe(df_train)
  train_classes = df_train[column_names.SPORT_UTILITY_VEHICLE].apply(get_prediction_name)
  temp = df_train.copy()
  df_train_prepared = prepare_for_prediction(df_train)
  data_train = Data(df_train_prepared, df_train, train_classes, total_train, su_train, other_train)

  df_real = pd.concat([df, temp]).drop_duplicates(keep=False)
  total_real, su_real, other_real = describe(df_real)
  real_classes = df_real[column_names.SPORT_UTILITY_VEHICLE].apply(get_prediction_name)
  df_real_prepared = prepare_for_prediction(df_real)
  data_real = Data(df_real_prepared, df_real, real_classes, total_real, su_real, other_real)

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
