from art import tprint
import pandas as pd
from enum import StrEnum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


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


def main():
  df = get_data()
  df_sport_utility_train = df[df[column_names.SPORT_UTILITY_VEHICLE] == True].sample(20)
  df_other_train = df[df[column_names.SPORT_UTILITY_VEHICLE] == False].sample(50)
  df_train = pd.concat([df_sport_utility_train, df_other_train])
  classes = df_train[column_names.SPORT_UTILITY_VEHICLE].apply(lambda v: 'SPORT_UTILITY' if v else 'OTHER')
  df_train_filtered = df_train.filter(items=column_names.train_columns())
  df_train_filtered.fillna(0, inplace=True)

  knn = KNeighborsClassifier(n_neighbors=7)
  knn_model = knn.fit(df_train_filtered, classes)
  # model accuracy test
  knn_accuracy_test_predictions = knn.predict(df_train_filtered)
  accuracy = accuracy_score(classes, knn_accuracy_test_predictions)
  print(f'Accuracy: {accuracy}')
  
  df_real_data = pd.concat([df, df_train]).drop_duplicates(keep=False)
  df_real_data_filtered = df_real_data.filter(items=column_names.train_columns())
  df_real_data_filtered.fillna(0, inplace=True)
  knn_prediction = knn.predict(df_real_data_filtered)
  df_real_data['prediction'] = knn_prediction
  print(f'Всего данных: {len(df_real_data)}')
  real_classes = df_real_data[column_names.SPORT_UTILITY_VEHICLE].apply(lambda v: 'SPORT_UTILITY' if v else 'OTHER')
  accuracy = accuracy_score(real_classes, knn_prediction)
  print(f'Точноcть на реальных данных: {accuracy}')


  # print(df_real_data[df_real_data['prediction'] == 'SPORT_UTILITY'].head(50))
  df_real_data.to_csv('output.csv', sep=';')


def get_data():
  df = pd.read_csv(SOURCE_FILE_NAME, sep=CSV_SEPARATOR, names=[v for v in column_names])
  df[column_names.VEHICLE_NAME] = df[column_names.VEHICLE_NAME].apply(lambda v: v.strip())
  for c in column_names.bool_columns():
    df[c] = df[c].astype(bool)
  return df

if __name__ == '__main__':
  tprint(TITLE)
  main()
