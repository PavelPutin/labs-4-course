from art import tprint
import pandas as pd
from enum import StrEnum


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
      column_names.ALL_WHEEL_DRIVE,
      column_names.REAR_WHEEL_DRIVE,
    )


def main():
  df = get_data()
  print(df.head())


def get_data():
  df = pd.read_csv(SOURCE_FILE_NAME, sep=CSV_SEPARATOR, names=[v for v in column_names])
  df[column_names.VEHICLE_NAME] = df[column_names.VEHICLE_NAME].apply(lambda v: v.strip())
  for c in column_names.bool_columns():
    df[c] = df[c].astype(bool)
  return df

if __name__ == '__main__':
  tprint(TITLE)
  main()
