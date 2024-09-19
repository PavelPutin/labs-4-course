import pandas as pd
from art import tprint
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


TITLE = 'Lab 2'
SOURCE_FILE = '..\\11_Вина Автралии fort\\wine_Austral.dat'
COLUMN_SEPARATOR = '\t'
COLUMNS = ('fort', 'date_')


def main():
  df = read_data()
  plot_data(df)


def plot_data(df):
  fig, ax = plt.subplots()
  ax.plot(df.get(COLUMNS[0]))
  ax.xaxis.set_major_locator(mticker.MultipleLocator(12))
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(get_date_ticks(df.get(COLUMNS[1]))))
  ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
  ax.xaxis.set_minor_formatter(mticker.NullFormatter())
  plt.show()


def read_data():
  return pd.read_csv(SOURCE_FILE, sep=COLUMN_SEPARATOR, usecols=COLUMNS, parse_dates=True)


def get_date_ticks(dates):
  def inner(x, pos):
    if 0 <= int(x) < len(dates) and int(x) % 12 == 0:
      return dates[int(x)]
    return ''
  return inner


if __name__ == '__main__':
  tprint(TITLE)
  main()
