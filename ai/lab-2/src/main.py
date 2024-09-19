import pandas as pd
from art import tprint
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import locale


TITLE = 'Lab 2'
SOURCE_FILE = '..\\11_Вина Автралии fort\\wine_Austral.dat'
COLUMN_SEPARATOR = '\t'
COLUMNS = ('fort', 'date_')
FORECAST_PERIODS = 8


def main():
  locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
  df = read_data()
  df_trend = df[COLUMNS[0]].rolling(window=12).mean()
  df_season = df[COLUMNS[0]].rolling(window=2).mean()
  df = make_forecast(df)
  plot_data(df, df_trend, df_season)


def make_forecast(df):
  forecast_column = COLUMNS[0]
  model = ARIMA(df[forecast_column], order=(1, 1, 1))
  model_fit = model.fit()
  forecast = model_fit.forecast(steps=FORECAST_PERIODS)

  future_dates = pd.date_range(start=df[COLUMNS[1]][len(df[COLUMNS[1]]) - 1] + pd.DateOffset(months=1), periods=FORECAST_PERIODS, freq='ME')
  forecast_df = pd.DataFrame({
    'date_': future_dates,
    'forecast': forecast
  })

  print('Прогнозируемые значения')
  for date, value in zip(forecast_df['date_'], forecast_df['forecast']):
    print(f'{date.strftime('%b %Y')}: {value:.3f}')

  df = df._append(forecast_df, ignore_index=True)
  return df


def plot_data(df, trend, season):
  fig, ax = plt.subplots()
  ax.plot(df[COLUMNS[0]][:-FORECAST_PERIODS], marker='o', linestyle='-', label='Продажи')
  ax.plot(df['forecast'][-FORECAST_PERIODS:], marker='o', label='Прогноз')
  ax.plot(trend, label='Тренд')
  ax.plot(season, label='Сезонность', linestyle='--')
  ax.xaxis.set_major_locator(mticker.MultipleLocator(12))
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(get_date_ticks(df.get(COLUMNS[1]))))
  ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
  ax.xaxis.set_minor_formatter(mticker.NullFormatter())
  ax.grid(True)
  ax.legend()
  plt.show()


def read_data():
  df = pd.read_csv(SOURCE_FILE, sep=COLUMN_SEPARATOR, usecols=COLUMNS, parse_dates=True)
  df[COLUMNS[1]] = pd.to_datetime(df[COLUMNS[1]], format='mixed')
  return df


def get_date_ticks(dates):
  def inner(x, pos):
    if 0 <= int(x) < len(dates):
      return dates[int(x)].strftime('%b %Y')
    return ''
  return inner


if __name__ == '__main__':
  tprint(TITLE)
  main()
