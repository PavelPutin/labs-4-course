from art import tprint
import pandas as pd
from enum import StrEnum


TITLE = 'Lab 3'
SOURCE_FILE_NAME = '..\\11_2004 New_Car_and_Truck_Data_SUV\\04cars.dat'
CSV_SEPARATOR = ';'
class column_names(StrEnum):
  VEHICLE_NAME = 'Vehicle Name'
  SPORTS_CAR = 'Sports Car'
  SPROT_UTILITY_VEHICLE = 'Sport Utility Vehicle'
  WAGON = 'Wagon'
  MINIVAN = 'Minivan'
  PICKUP = 'Pickup'
  ALL_WHEEL_DRIVE = 'All-Wheel Drive'
  REAR_WHEEL_DRIVE = 'Rear-Wheel Drive'
  SUGGESTED_RETAIL_PRIC = 'Suggested Retail Pric'
  DEALER_COST = 'Dealer Cost',
  # 'Engine Size',
  # 'Number of Cylinders',
  # 'Horsepower',
  # 'City Miles Per Gallon',
  # 'Highway Miles Per Gallon',
  # 'Weight',
  # 'Wheel Base',
  # 'Length',
  # 'Width',



def main():
  df = pd.read_csv(SOURCE_FILE_NAME, sep=CSV_SEPARATOR, names=[v for v in column_names])
  print(df.head())

if __name__ == '__main__':
  tprint(TITLE)
  main()
