# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 00:42:31 2022

@author: Alexander A. Nazarov
"""


###############################################################################
#           СТАТИСТИЧЕСКИЙ АНАЛИЗ / STATISTICAL ANALYSIS
#------------------------------------------------------------------------------
#           ПОЛЬЗОВАТЕЛЬСКИЕ СТАТИСТИЧЕСКИЕ ФУНКЦИИ
#           CUSTOM STATISTICAL FUNCTIONS
###############################################################################

import time
start_time = time.time()

#%clear    # команда очистки консоли


CALCULATION_VERSION = 220709    # ВЕРСИЯ РАСЧЕТА


#==============================================================================
#               ПОДКЛЮЧЕНИЕ МОДУЛЕЙ И БИБЛИОТЕК
#==============================================================================

import os
import sys
import platform

import math    # модуль доступа к математическим функциям
from math import *    # подключаем все содержимое модуля math, используем без псевдонимов

import numpy as np
from numpy import nan

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import scipy as sci
import scipy.stats as sps

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

import statistics as stat    # module 'statistics' has no attribute '__version__'

import sympy as sym


#=============================================================================
#               КОНСТАНТЫ
#=============================================================================

INCH = 25.4                                                                     # мм/дюйм
NUMBER_CHAR_LINE = 79    # число символов в строке
TABLE_HEAD_1 = ('#' * NUMBER_CHAR_LINE)
TABLE_HEAD_2 = ('=' * NUMBER_CHAR_LINE)
TABLE_HEAD_3 = ('-' * NUMBER_CHAR_LINE)


#=============================================================================
#               НАСТРОЙКА ШРИФТОВ В ГРАФИКАХ
#=============================================================================

f_size = 8    # пользовательская переменная для задания базового размера шрифта



#=============================================================================
#               СОДЕРЖАНИЕ
#=============================================================================

#------------------------------------------------------------------------------
#   1. ФУНКЦИИ ДЛЯ ПЕРВИЧНОЙ ОБРАБОТКИ ДАННЫХ:
#    
#       df_system_memory
#       unique_values
#       transformation_to_category
#       df_detection_values
#
#
#   2. ФУНКЦИИ ДЛЯ ОПИСАТЕЛЬНОЙ СТАТИСТИКИ:
#
#       descriptive_characteristics
#       descriptive_characteristics_df_group
#
#
#   3. ФУНКЦИИ ДЛЯ ОЦЕНКИ РАЗЛИЧНЫХ ПАРАМЕТРОВ
#
#       confidence_interval_ratio
#
#
#   4. ФУНКЦИИ ДЛЯ ПРОВЕРКИ РАЗЛИЧНЫХ ГИПОТЕЗ
#
#       KS_ES_2samp_test
#       Mann_Whitney_test
#       Wilcoxon_test
#       Kruskal_Wallis_test
#       Ansari_Bradley_test
#       Mood_test
#       Levene_test
#       Fligner_Killeen_test
#
#
#   5. ФУНКЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ
#
#       graph_hist_boxplot_probplot_sns
#       graph_hist_boxplot_probplot_XY_sns
#       graph_ecdf_cdf_mpl
#       graph_lorenz_curve
#       graph_df_heatmap
#
#
#   6. ФУНКЦИИ ДЛЯ ИССЛЕДОВАНИЯ ЗАКОНОВ РАСПРЕДЕЛЕНИЯ
#
#       norm_distr_check
#
#
#   7. ФУНКЦИИ ДЛЯ ВЫЯВЛЕНИЯ АНОМАЛЬНЫХ ЗНАЧЕНИЙ (ВЫБРОСОВ)
#
#       detecting_outliers_mad_test
#
#
#   8. ФУНКЦИИ ДЛЯ ИССЛЕДОВАНИЯ КАТЕГОРИАЛЬНЫХ ДАННЫХ
#
#       graph_contingency_tables_hist_3D
#       graph_contingency_tables_mosaicplot_sm
#       make_color_mosaicplot_dict
#       graph_contingency_tables_bar_freqint
#       graph_contingency_tables_heatmap
#       conjugacy_table_independence_check
#
#
#   9. ФУНКЦИИ ДЛЯ КОРРЕЛЯЦИОННОГО АНАЛИЗА
#
#       Cheddock_scale_check
#       Evans_scale_check
#       Rea_Parker_scale_check
#
#       rank_corr_coef_check
#
#
#
#
#   СТАРЫЕ ФУНКЦИИ
#
#------------------------------------------------------------------------------


#==============================================================================
#               1. ФУНКЦИИ ДЛЯ ПЕРВИЧНОЙ ОБРАБОТКИ ДАННЫХ
#==============================================================================

'''print(TABLE_HEAD_2)
Title = "1. ФУНКЦИИ ДЛЯ ПЕРВИЧНОЙ ОБРАБОТКИ ДАННЫХ"
print(' ' * int((NUMBER_CHAR_LINE - len(Title))/2), Title)
print(TABLE_HEAD_2, 2*'\n')'''


#------------------------------------------------------------------------------
#   Функция df_system_memory
#
#   Возвращает объем памяти, занимаемой DataFrame (на системном уровне)
#------------------------------------------------------------------------------

def df_system_memory(
    df_in:           pd.core.frame.DataFrame,
    measure_unit:    str = 'MB',
    digit:           int = 2,
    detailed:        bool = True):
    
    """
    Возвращает объем памяти, занимаемой DataFrame (на системном уровне)

    Args:
        df_in (pd.core.frame.DataFrame):    
            массив исходных данных.
        
        measure_unit (str, optional):    
            вид единиц измерения ('MB' - мБ, 'B' - байт). 
            Defaults to 'MB'.
        
        digit (int, optional):        
            точность выводимого результата. 
            Defaults to 2.
        
        detailed (bool, optional):         
            логический параметр: True/False - выводить/не выводить детализацию 
            по столбцам датасета). 
            Defaults to True.

    Returns:    
        result (float):
            объем памяти, занимаемой DataFrame (на системном уровне)
            
    """
    
    df_in_memory_usage = df_in.memory_usage(deep=True)
    if detailed:
        display(df_in_memory_usage)        
    
    df_in_memory_usage_sum = df_in_memory_usage.sum()
    if measure_unit=='MB':
        result = round(df_in_memory_usage_sum / (2**20), digit)
        
    elif measure_unit=='B':
        result = round(df_in_memory_usage_sum, 0)

    return result 
    
    

#------------------------------------------------------------------------------
#   Функция unique_values
#
#   Возвращает число уникальных значений, тип данных и объем занимаемой памяти 
#   по столбцам исходного датасета
#------------------------------------------------------------------------------

def unique_values(
    df_in:      pd.core.frame.DataFrame,
    sorting:    bool = True):
    
    """
    Возвращает число уникальных значений, тип данных и объем занимаемой памяти 
    по столбцам исходного датасета.

    Args:
        df_in (pd.core.frame.DataFrame):
            массив исходных данных.
            
        sorting (bool, optional):
            логический параметр: True/False - сортировать/не сортировать 
            признаки по степени убывания уникальных значений).
            Defaults to True.
    
    Returns:    
        df_out (pd.core.frame.DataFrame):
            результат            
            
    Notes:
        Принимает в качестве аргумента DataFrame и возвращает другой 
        DataFrame со следующей структурой:
            - индекс - наименования столбцов из исходного DataFrame
            - столбец Num_Unique - число уникальных значений
            - столбец Type - тип данных
            - столбец Memory_usage (MB) - объем занимаемой памяти на системном 
                уровне (в МБ)
            
    """
    
    df_out = pd.DataFrame(
        np.transpose(
            [df_in.nunique(),
             df_in.dtypes,
             round(df_in.memory_usage(
                 deep=True).iloc[1:len(df_in)] / (2**20), 2)]),
        index=df_in.columns,
        columns=['Num_Unique', 'Type', 'Memory_usage (MB)'])
    
    if sorting:
        return df_out.sort_values(by='Num_Unique', ignore_index=False)
    else:
        return df_out 



#------------------------------------------------------------------------------
#   Функция transformation_to_category
#
#   Принимает в качестве аргумента DataFrame и преобразует выбранные 
#   признаки в категориальные
#------------------------------------------------------------------------------  

def transformation_to_category(
    data_df:             pd.core.frame.DataFrame,
    max_unique_count:    int = 150, 
    cols_to_exclude:     list = None):
    
    """
    Принимает в качестве аргумента DataFrame и преобразует выбранные признаки 
    в категориальные.

    Args:
        df_in (pd.core.frame.DataFrame):     
            массив исходных данных.
            
        max_unique_count (int, optional):    
            максимальное число уникальных значений, при котором признак 
            преобразуется в категориальный. 
            Defaults to 150.
            
        cols_to_exclude (list, optional):    
            список-исключение признаков, которые не преобразуются 
            в категориальные. 
            Defaults to None.
    
    
    Returns:    
        data_df (pd.core.frame.DataFrame):
            преобразованный DataFrame            
            
    """
    
    # количество уникальных значений по столбцам исходной таблицы
    df_number_unique_values = unique_values(data_df, sorting=False)    
    
    # особенность аргумента по умолчанию, который является списком (list)
    cols_to_exclude = cols_to_exclude or []    
    
    for col in data_df.columns:
        if ((data_df[col].nunique() < max_unique_count) and 
            (col not in cols_to_exclude)):    
                data_df[col] = data_df[col].astype('category')    

    return data_df   

    

#------------------------------------------------------------------------------
#   Функция df_detection_values
#
#   Визуализация пропусков в датасете с помощью графика тепловой карты
#------------------------------------------------------------------------------

def df_detection_values(
    df_in:               pd.core.frame.DataFrame,
    detection_values:    list = [nan, pd.NA, 0],
    color_highlight:     str = 'yellow',
    graph_size:          tuple = (210/INCH, 297/INCH/2)):
    
    """
    Визуализация пропусков в датасете с помощью графика тепловой карты 
    (heatmap).

    Args:
        df_in (pd.core.frame.DataFrame):      
            массив исходных данных.
            
        detection_values (list, optional):    
            список значений, которые могут рассматриваться как пропуски. 
            Defaults to [nan, pd.NA, 0].
            
        color_highlight (str, optional):      
            цвет для маркирования пропусков на графике тепловой карты. 
            Defaults to 'yellow'.
            
        graph_size (tuple, optional):         
            размер графика в дюймах. 
            Defaults to (210/INCH, 297/INCH/2), 
                где константа INCH = 25.4 мм/дюйм
    
    Returns:
        result_df (pd.core.frame.DataFrame):        
            таблица (DataFrame) с числом пропусков по признакам исходного 
            датасета. 
            
        detection_values_df (pd.core.frame.DataFrame):    
            таблица (DataFrame), имеющая структуру исходного датасета,
            в которой пропуски маркированы как True, 
            а отсутствие пропуска - False.
            
    """
    
    cols = df_in.columns
    detection_values_df = df_in.isin(detection_values)
    
    col_size = np.array([df_in[elem].size for elem in cols])
    col_detection_values_count = [detection_values_df[elem].sum() for elem in cols]
        
    result_df = pd.DataFrame(
        {'size': col_size,
         'detected values': col_detection_values_count,
         'percentage of detected values': col_detection_values_count/col_size},
        index=df_in.columns)
    
    fig, axes = plt.subplots(figsize=graph_size)
    sns.heatmap(
        detection_values_df,
        cmap=sns.color_palette(['grey', color_highlight]),
        cbar=False,
        ax=axes)
    plt.show()
       
    return result_df, detection_values_df




#==============================================================================
#               2. ФУНКЦИИ ДЛЯ ОПИСАТЕЛЬНОЙ СТАТИСТИКИ
#==============================================================================


#------------------------------------------------------------------------------
#   Функция descriptive_characteristics
#
#   Принимает в качестве аргумента np.ndarray и возвращает DataFrame,
#   содержащий основные статистические характеристики с доверительными 
#   интервалами и ошибками определения - расширенный аналог describe
#------------------------------------------------------------------------------

def descriptive_characteristics(
    X,
    p_level:            float = 0.95,
    auxiliary_table:    bool = False):
    
    """
    Принимает в качестве аргумента np.ndarray и возвращает DataFrame,
    содержащий основные статистические характеристики с доверительными 
    интервалами и ошибками определения - расширенный аналог describe.

    Args:
        X:                       
            мсходный массив данных
            
        p_level (float, optional):           
            доверительная вероятность. 
            Defaults to 0.95.
            
        auxiliary_table (bool, optional):    
            логический параметр: True/False - возвращать/не возвращать таблицу 
            со вспомогательными значениями.
            Defaults to False.
    
    Returns:
        result (pd.core.frame.DataFrame):
            результат            
        
        result_auxiliary (pd.core.frame.DataFrame):
            таблица со вспомогательными значениями
                    
    """
    
    a_level = 1 - p_level
    
    # Вспомогательные величины
    #-------------------------
    
    # распределение Стьюдента
    f_t = len(X)-1    # число степеней свободы для t-квантиля
    gamma_t = 1 - a_level/2    # довер.вер-ть для t-квантиля
    t_p = sps.t.ppf(gamma_t, f_t)    # табл.значение t-квантиля
    
    # нормальное распределение
    u_p = sps.norm.ppf(p_level, 0, 1)    # табл.значение квантиля норм.распр.
    
    # распределение хи-квадрат
    f_chi2 = len(X) - 1    # число степеней свободы хи2
    gamma_chi2_low = (1 + p_level)/2    # довер.вер-ть хи2 нижн.
    gamma_chi2_up = (1 - p_level)/2    # довер.вер-ть хи2 верх.
    chi2_p_low = sps.chi2.ppf(gamma_chi2_low, f_chi2)
    chi2_p_up = sps.chi2.ppf(gamma_chi2_up, f_chi2)
    
    # Расчет статистических характеристик
    #------------------------------------
    
    # объем выборки
    N = round(len(X), 0)
    
    # среднее арифметическое
    X_mean = X.mean()
    X_std = X.std(ddof = 1)
    conf_int_low_X_mean = X_mean - sps.t.ppf(gamma_t, f_t) * X_std/math.sqrt(N)
    conf_int_up_X_mean = X_mean + sps.t.ppf(gamma_t, f_t) * X_std/math.sqrt(N)
    abs_err_X_mean = X_std / math.sqrt(N)
    rel_err_X_mean = abs_err_X_mean / X_mean * 100
    
    # медиана
    Me = np.median(X)
    C_p_Me = floor((N +  1 - u_p*sqrt(N + 0.5 - 0.25*u_p**2))/2) # вспом.величина
    X_sort = np.array(sorted(X))
    conf_int_low_Me = X_sort[C_p_Me-1]
    conf_int_up_Me = X_sort[(N - C_p_Me + 1) - 1]
    abs_err_Me = abs_err_X_mean * sqrt(pi/2)
    rel_err_Me = abs_err_Me / Me * 100
    # довер.интервал для медианы - см. ГОСТ Р 50779.24-2005, п.6.2-6.3
    if X_mean < Me:
        Me_note = 'distribution is negative skewed \
            (левосторонняя асимметрия) (mean < median)'
    else:
        Me_note = 'distribution is positive skewed \
            (правосторонняя асимметрия) (mean > median)'
    
    # мода        
    Mo = stat.mode(X)
    
    # выборочная дисперсия
    D_X = np.var(X, ddof = 1)
    conf_int_low_D_X = D_X * (N - 1)/chi2_p_low
    conf_int_up_D_X = D_X * (N - 1)/chi2_p_up
    abs_err_D_X = D_X / sqrt(2*N)
    rel_err_D_X = abs_err_D_X / D_X * 100
    
    # выборочное С.К.О.
    conf_int_low_X_std = sqrt(conf_int_low_D_X)
    conf_int_up_X_std = sqrt(conf_int_up_D_X)
    abs_err_X_std = X_std / sqrt(2*N)
    rel_err_X_std = abs_err_X_std / X_std * 100   
    
    # выборочное С.Л.О.
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    # также смотри 
    # https://www.statsmodels.org/stable/generated/statsmodels.robust.scale.mad.html
    #mad_X = pd.Series(X).mad()
    mad_X = sps.median_abs_deviation(X)
        
    # минимальное, максимальное значения и вариационный размах
    min_X = np.amin(X)
    max_X = np.amax(X)
    R = np.ptp(X)
    
    # выборочные квантили
    np.percentile(X, 25)
    np.percentile(X, 50)
    np.percentile(X, 75)
    IQR = np.percentile(X, 75) - np.percentile(X, 25)
    np.percentile(X, 5)
    np.percentile(X, 95)
    
    # коэффициент вариации
    CV = sps.variation(X)
    conf_int_low_CV = CV / (1 + u_p/sqrt(2*(N - 1)) * sqrt(1 + 2*CV**2))
    conf_int_up_CV = CV / (1 - u_p/sqrt(2*(N - 1)) * sqrt(1 + 2*CV**2))
    abs_err_CV = CV / sqrt(N - 1) * sqrt(0.5 + CV**2)
    rel_err_CV = abs_err_CV / CV * 100
    if CV <= 0.33:
        CV_note = 'CV <= 0.33 (homogeneous population)'
    else:
        CV_note = 'CV > 0.33 (heterogeneous population)'
    
    # квартильный коэффициент дисперсии
    QCD = (np.percentile(X, 75) - np.percentile(X, 25)) / (np.percentile(X, 75) 
        + np.percentile(X, 25))
    
    # показатель асимметрии
    As = sps.skew(X)
    abs_err_As = sqrt(6*N*(N-1) / ((N-2)*(N+1)*(N+3)))
    rel_err_As = abs_err_As / As * 100
    
    if abs(As) <= 0.25:
        As_note = 'distribution is approximately symmetric \
            (распределение приблизительно симметричное) (abs(As)<=0.25)'
    elif abs(As) <= 0.5:
        if As < 0:
            As_note = 'distribution is moderately negative skewed \
                (умеренная левосторонняя асимметрия) (abs(As)<=0.5, As<0)'
        else:
            As_note = 'distribution is moderately positive skewed \
                (умеренная правосторонняя асимметрия) (abs(As)<=0.5, As>0)'
    else:
        if As < 0:
            As_note = 'distribution is highly negative skewed \
                (значительная левосторонняя асимметрия) (abs(As)>0.5, As<0)'
        else:
            As_note = 'distribution is highly positive skewed \
                (значительная правосторонняя асимметрия) (abs(As)>0.5, As>0)'
            
    # показатель эксцесса
    Es = sps.kurtosis(X)
    abs_err_Es = sqrt(24*N*(N-1)**2 / ((N-3)*(N-2)*(N+3)*(N+5)))
    rel_err_Es = abs_err_Es / Es * 100
    if Es > 0:
        Es_note = 'leptokurtic distribution (островершинное распределение) \
            (Es>0)'
    elif Es < 0:
        Es_note = 'platykurtic distribution (плосковершинное распределение) \
            (Es<0)'
    else:
        Es_note = 'mesokurtic distribution (нормальное распределение) (Es=0)'
    
    # Создадим DataFrame для сводки результатов
    #------------------------------------------
    
    # основная таблица
    result = pd.DataFrame({
        'characteristic': (
            'count', 'mean', 'median', 'mode',
            'variance', 'standard deviation', 'mean absolute deviation',
            'min', '5%', '25% (Q1)', '50% (median)', '75% (Q3)', '95%', 'max',
            'range = max − min', 'IQR = Q3 - Q1', 'CV = std/mean', 
            'QCD = (Q3-Q1)/(Q3+Q1)',
            'skew (As)', 'kurtosis (Es)'),
        'evaluation': (
            N, X_mean, Me, Mo,
            D_X, X_std, mad_X,
            min_X,
            np.percentile(X, 5), np.percentile(X, 25), np.percentile(X, 50), 
            np.percentile(X, 75), np.percentile(X, 95),
            max_X,
            R, IQR, CV, QCD,
            As, Es),
        'conf.int.low': (
            '', conf_int_low_X_mean, conf_int_low_Me, '',

            conf_int_low_D_X,  conf_int_low_X_std, '',
            '', '', '', '', '', '', '',
            '', '', conf_int_low_CV, '',
            '', ''),
        'conf.int.high': (
            '', conf_int_up_X_mean, conf_int_up_Me, '',
            conf_int_up_D_X, conf_int_up_X_std,
            '', '', '', '', '', '', '', '',
            '', '', conf_int_up_CV, '',
            '', ''),
        'abs.err.': (
            '', abs_err_X_mean, abs_err_Me, '',
            abs_err_D_X, abs_err_X_std,
            '', '', '', '', '', '', '', '',
            '', '', abs_err_CV, '',
            abs_err_As, abs_err_Es),
        'rel.err.(%)': (
            '', rel_err_X_mean, rel_err_Me, '',
            rel_err_D_X, rel_err_X_std,
            '', '', '', '', '', '', '', '',
            '', '', rel_err_CV, '',
            rel_err_As, rel_err_Es),
        'note': (
            '', '', Me_note, '', '', '', '', '', '', '', '', '', '', '', '', 
            '', CV_note, '',
            As_note, Es_note)
        })
            
    # вспомогательная таблица
    result_auxiliary = pd.DataFrame({
        'characteristic': (
            'confidence probability',
            'significance level',
            'count',
            "quantile of the Student's distribution",
            'quantile of the normal distribution',
            'quantile of the chi-square distribution (low)',
            'quantile of the chi-square distribution (up)'),
        'designation, formula': (
            'p',
            'a = 1 - p',
            'n',
            't(1 - a/2, n-1)',
            'u(p)',
            'chi2((1+p)/2, n-1)',
            'chi2((1-p)/2, n-1)'),
        'evaluation': (
            p_level, a_level, N, t_p, u_p, chi2_p_low, chi2_p_up)}
        )
    
    # вывод результатов
    if auxiliary_table:
        return result, result_auxiliary
    else:
        return result
    
    
    
#------------------------------------------------------------------------------
#   Функция descriptive_characteristics_df_group
#   
#   Получение дескриптивной статистики в разрезе категорий некоего 
#   качественного признака: функция принимает в качестве аргумента DataFrame, 
#   выполняет группировку по выбранному качественному признаку и рассчитывает 
#   основные дескриптивные показатели выбранного количественного признака 
#   в разрезе качественного признака.
#------------------------------------------------------------------------------

def descriptive_characteristics_df_group(
    data_df:                  pd.core.frame.DataFrame,
    column_qualitative:       str,
    column_quantitative:      str):
    
    """
    Получение дескриптивной статистики в разрезе категорий некоего 
    качественного признака: функция принимает в качестве аргумента DataFrame, 
    выполняет группировку по выбранному качественному признаку и рассчитывает 
    основные дескриптивные показатели выбранного количественного признака 
    в разрезе качественного признака.

    Args:
        data_df (pd.core.frame.DataFrame):           
            массив исходных данных.
            
        column_qualitative (str):                    
            имя количественного признака в DataFrame.
            
        column_quantitative (str):                   
            имя качественного признака в DataFrame.
            
    Returns:
        result (pd.core.frame.DataFrame):
            датасет со статистическими характеристиками
            
    """
    
    # создаем отдельные DataFrame, отфильтрованный по качественному признаку, 
    # и сохраняем их в словарь (dict)
    #column_quantitative_index_list = list(data_df[column_quantitative].unique())
    column_quantitative_index_list = \
        data_df[column_quantitative].unique().tolist()
    data_df_group_dict = dict()
    for elem in column_quantitative_index_list:
        mask_temp = data_df[column_quantitative] == elem
        data_group_temp = pd.DataFrame(data_df[mask_temp])
        #display(data_group_temp)
        data_df_group_dict[elem] = data_group_temp
   
    # формируем DataFrame с дескриптивными характеристиками 
    agg_list = ['sum', 'describe', 'median', 'skew']
    agg_list.remove('sum')
    data_group = \
        data_df.groupby(column_quantitative)[column_qualitative].agg(agg_list)
    
    # добавляем в DataFrame дескриптивные характеристики, отсутствующие в agg
    for row in data_group.index:    
        # https://en.wikipedia.org/wiki/Median_absolute_deviation
        data_group.loc[row, 'MAD'] = \
            sm.robust.scale.mad(\
                data_df_group_dict[row][column_qualitative], c=1)
        data_group.loc[row, 'range'] = data_group.loc[row]['describe']['max'] \
            - data_group.loc[row]['describe']['min']
        data_group.loc[row, 'IQR'] = data_group.loc[row][('describe', '75%')] \
            - data_group.loc[row][('describe', '25%')]
        data_group.loc[row, 'CV'] = data_group.loc[row]['describe']['std'] \
            / data_group.loc[row]['describe']['mean']
        data_group.loc[row, 'kurtosis'] = \
            sps.kurtosis(data_df_group_dict[row][column_qualitative])
    #display(data_group.style.format(precision=3, thousands=" "))

    # изменяем порядок столбцов в DataFrame
    current_order_new = [
        ('describe', 'count'),                
        ('describe', 'mean'), ('median', 'Balance'),                  
        ('describe', 'std'), ('MAD', ''),                 
        ('describe', 'min'), ('describe', '25%'), ('describe', '50%'), 
        ('describe', '75%'), ('describe', 'max'), 
        ('range', ''), ('IQR', ''),                  
        ('CV', ''), ('skew', 'Balance'), ('kurtosis', '')]
    data_group = data_group.reindex(columns = current_order_new)
    #display(data_group.style.format(precision=3, thousands=" "))

    # удаляем мультииндекс и изменяем названия столбцов 
    data_group.columns = data_group.columns.droplevel(0)
    new_columns_list = ['count', 'mean', 'median', 'std', 'MAD', 'min', \
        '25%', '50%', '75%', 'max', 'range', 'IQR', 'CV', 'skew', 'kurtosis']
    data_group.columns = new_columns_list
        
    result = data_group
    return result



#==============================================================================
#               3. ФУНКЦИИ ДЛЯ ОЦЕНКИ РАЗЛИЧНЫХ ПАРАМЕТРОВ
#==============================================================================

#------------------------------------------------------------------------------
#   Функция confidence_interval_ratio
#
#   Возвращает доверительный интервал доли p = m/n в виде списка [p_min, p_max]
#------------------------------------------------------------------------------

def confidence_interval_ratio(
    n:          int, 
    m:          int, 
    p_level:    float = 0.95, 
    digit:      int = 4):
    
    """
    Возвращает доверительный интервал доли p = m/n в виде списка [p_min, p_max]

    Args:
        n (int, optional):            
            общее число испытаний
            
        m (int, optional):            
            число испытаний, в которых наблюдалось рассматриваемое событие
            
        p_level (float, optional):    
            доверительная вероятность. 
            Defaults to 0.95
            
        digit (int, optional):        
            точность выдаваемого результата. 
            Defaults to 4
            
    Returns:
        [p_min, p_max] (list):
            доверительный интервал доли
        
    """
    
    q = m/n
    k1 = lambda q: log(1/(1-q))
    k2 = lambda n, m: m / sum([1/(n-i) for i in range(0, m)])
    
    if m <= 0.5*n:
        p_min = round(1-exp(-sps.chi2.ppf(1 - \
            (1 + p_level)/2, 2*m) / (2*k2(n, m))), digit)
        p_max = round(1-exp(-sps.chi2.ppf(1 - 
            (1 - p_level)/2, 2*(m+1)) / (2*k2(n, m+1))), digit)
    else:
        p_min = round(exp(-sps.chi2.ppf(1 - \
            (1 - p_level)/2, 2*(n-m+1))/ (2*k2(n, n-m+1))), digit)
        p_max = round(exp(-sps.chi2.ppf(1 - \
            (1 + p_level)/2, 2*(n-m))/ (2*k2(n, n-m))), digit)
        
    return [p_min, p_max]




#==============================================================================
#               4. ФУНКЦИИ ДЛЯ ПРОВЕРКИ РАЗЛИЧНЫХ ГИПОТЕЗ
#==============================================================================


#------------------------------------------------------------------------------
#   Функция KS_ES_2samp_test
#   
#   Функция сравнивает основные распределения двух независимых выборок: 
#    - по критерию Колмогорова-Смирнова
#    - по критерию Эппса-Синглтона ES.
#------------------------------------------------------------------------------

def KS_ES_2samp_test(
    data1, data2,
    alternative:    str = 'two-sided',
    method:         str = 'auto',    
    p_level:        float = 0.95):
    
    """
    Функция сравнивает основные распределения F(x) и G(x) двух независимых выборок: 
        - по критерию Колмогорова-Смирнова KS (только для непрерывных распределений);
        - по критерию Эппса-Синглтона ES.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html#scipy.stats.ks_2samp
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.epps_singleton_2samp.html
        
    Args:
        data1, data2:                       
            исходные массивы данных. Должны иметь одинаковый размер.
            
        alternative (str, optional):        
            альтернативная гипотеза {‘two-sided’, ‘less’, ‘greater’}.                                            
            Defaults to ‘two-sided’.
            
        method (str, optional):             
            метод расчета уровня значимости {‘auto’, ‘asymptotic’, ‘exact’}                                             
            Defaults to ‘auto’.
            
        p_level (float, optional):          
            доверительная вероятность. 
            Defaults to 0.95.
            
    Returns:
        result (pd.core.frame.DataFrame):
            результат.
            
    """ 
    
    a_level = 1 - p_level
    
    # реализация теста Колмогорова-Смирнова KS
    test_KS = sps.ks_2samp(
        data1, data2,
        alternative = alternative,
        method = method)
    s_KS_calc = test_KS.statistic
    a_KS_calc = test_KS.pvalue
    conclusion_KS = 'H0: the data were drawn from the same distribution' \
        if a_KS_calc >= a_level else \
        'H1: the data were not drawn from the same distribution'
    
    # реализация теста Эппса-Синглтона ES
    test_ES = sps.epps_singleton_2samp(
        data1, data2)
    s_ES_calc = test_ES.statistic
    a_ES_calc = test_ES.pvalue
    conclusion_ES = 'H0: the data were drawn from the same distribution' \
        if a_ES_calc >= a_level else \
        'H1: the data were not drawn from the same distribution'
    
    # DataFrame для сводки результатов    
    result = pd.DataFrame({
        'test': ('Kolmogorov-Smirnov test (KS)', 'Epps-Singleton test (ES)'),
        'p_level': (p_level),
        'a_level': (a_level),
        #'null hypothesis': ('H0: the data were drawn from the same distribution'),
        #'alternative hypothesis': ('H1: the data were not drawn from the same distribution'),
        'a_calc': (a_KS_calc, a_ES_calc),
        'a_calc >= a_level': (a_KS_calc >= a_level, a_ES_calc >= a_level),
        'statistic': (s_KS_calc, s_ES_calc),
        #'critical_value': ('-'),
        #'statistic <= critical_value': ('-'),
        #'hypothesis check': (''),
        'conclusion (accepted hypothesis)': (conclusion_KS, conclusion_ES),
    })
    
    return result



#------------------------------------------------------------------------------
#   Функция Mann_Whitney_test
#   
#   U-критерий Манна-Уитни
#------------------------------------------------------------------------------

def Mann_Whitney_test(
    data1, data2,                   
    use_continuity:    bool = True,    
    alternative:       str = 'two-sided',
    method:            str = 'auto',    
    p_level:           float = 0.95):
    
    """
    Функция выполняет по U-критерий Манна-Уитни непараметрическую проверку 
    нулевой гипотезы о том, что распределение, лежащее в основе выборки X , 
    такое же, как распределение, лежащее в основе выборки Y.
    Его часто используют как тест на разницу в местоположении между 
    дистрибутивами.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu
        
    Args:
        data1, data2:                       
            исходные массивы данных.
            
        use_continuity (bool, optional):    
            логический параметр: True/False - использовать/не использовать 
            поправку на непрерывность.                                             
            Defaults to True.
            
        alternative (str, optional):        а
        льтернативная гипотеза {“two-sided”, “greater”, “less”}.                                            
        Defaults to ‘two-sided’.
        
        method (str, optional):             
            метод расчета уровня значимости {‘auto’, ‘asymptotic’, ‘exact’}                                                                                      
            Defaults to ‘auto’.
        
        p_level (float, optional):          
            доверительная вероятность. 
            Defaults to 0.95

    Returns:
        result (pd.core.frame.DataFrame):
            результат.
            
    """    
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sps.mannwhitneyu(
        data1, data2,
        use_continuity = use_continuity,
        alternative = alternative,
        method = method)
    s_calc = test.statistic    # расчетное значение статистики критерия
    a_calc = test.pvalue    # расчетный уровень значимости
    #hypothesis_check = 'H0' if a_calc >= a_level else 'H1'
    conclusion = 'H0: The sample distributions are equal' if a_calc >= a_level \
        else 'H1: The sample distributions are not equal'
    
    # DataFrame для сводки результатов    
    result = pd.DataFrame({
        'test': ('Mann-Whitney test'),
        'p_level': (p_level),
        'a_level': (a_level),
        #'null hypothesis': ('H0: The sample distributions are equal'),
        #'alternative hypothesis': ('H1: The sample distributions are not equal'),
        'a_calc': (a_calc),
        'a_calc >= a_level': (a_calc >= a_level),
        'statistic': (s_calc),
        #'critical_value': ('-'),
        #'statistic <= critical_value': ('-'),
        #'hypothesis check': (''),
        'conclusion (accepted hypothesis)': (conclusion),
        },
        index=[0])
    
    return result



#------------------------------------------------------------------------------
#   Функция Wilcoxon_test
#   
#   Критерий Уилкоксона
#------------------------------------------------------------------------------

def Wilcoxon_test(
    data1, data2,
    zero_method:    str = 'wilcox',                       
    correction:     bool = True,    
    alternative:    str = 'two-sided',
    method:         str = 'auto',    
    p_level:        float = 0.95):
    
    """
    Функция выполняет проверку нулевую гипотезу о том, что две связанные 
    парные выборки происходят из одного и того же распределения. 
    В частности, проверяет, является ли распределение различий симметричным 
    относительно нуля. Это непараметрическая версия парного Т-теста.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#scipy.stats.wilcoxon
        
    Args:
        data1, data2:                       
            исходные массивы данных. Должны иметь одинаковый размер.
            
        zero_method (str, optional):        
            метод обработки пар наблюдений с одинаковыми значениями 
            («нулевые разности» или «нули»)   
            {“wilcox”, “pratt”, “zsplit”}      
            Defaults to “wilcox”.
            
        correction (bool, optional):        
            логический параметр: True/False - использовать/не использовать 
            поправку на непрерывность.    
            Defaults to True.
            
        alternative (str, optional):        
            альтернативная гипотеза {‘two-sided’, ‘less’, ‘greater’}.                                            
            Defaults to ‘two-sided’.
            
        method (str, optional):             
            метод расчета уровня значимости {‘auto’, ‘asymptotic’, ‘exact’}                                          
            Defaults to ‘auto’.
            
        p_level (float, optional):          
            доверительная вероятность. Defaults to 0.95.
    
    Returns:
        result (pd.core.frame.DataFrame):
            результат.
            
    """    
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sps.wilcoxon(
        data1, data2,
        zero_method = zero_method,                       
        correction = correction,    
        alternative = alternative,
        method = method)
    s_calc = test.statistic    # расчетное значение статистики критерия
    a_calc = test.pvalue    # расчетный уровень значимости
    #hypothesis_check = 'H0' if a_calc >= a_level else 'H1'
    conclusion = 'H0: The sample distributions are equal' if a_calc >= a_level else 'H1: The sample distributions are not equal'
    
    # DataFrame для сводки результатов    
    result = pd.DataFrame({
        'test': ('wilcoxon test'),
        'p_level': (p_level),
        'a_level': (a_level),
        #'null hypothesis': ('H0: The sample distributions are equal'),
        #'alternative hypothesis': ('H1: The sample distributions are not equal'),
        'a_calc': (a_calc),
        'a_calc >= a_level': (a_calc >= a_level),
        'statistic': (s_calc),
        #'critical_value': ('-'),
        #'statistic <= critical_value': ('-'),
        #'hypothesis check': (''),
        'conclusion (accepted hypothesis)': (conclusion),
        },
        index=[0])
    
    return result



#------------------------------------------------------------------------------
#   Функция Kruskal_Wallis_test
#   
#   Критерий Краскела-Уоллиса
#------------------------------------------------------------------------------

def Kruskal_Wallis_test(
    *samples,
    p_level:    float = 0.95):
    
    """
    H-тест Краскела-Уоллеса проверяет нулевую гипотезу о том, что медиана 
    популяции всех групп равна. Это непараметрическая версия ANOVA.
    Тест работает на двух и более независимых выборках, которые могут иметь 
    разные размеры.
    Обратите внимание, что отказ от нулевой гипотезы не указывает на то, 
    какая из групп отличается.
    Апостериорные сравнения между группами необходимы, чтобы определить, 
    какие группы отличаются.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html#scipy.stats.kruskal
        
    Args:
        samples:                      
            двумерный массив, содержащий исходные выборки.
            
        p_level (float, optional):    
            доверительная вероятность. 
            Defaults to 0.95.

    Returns:
        result (pd.core.frame.DataFrame):
            результат.
            
    """ 
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sps.kruskal(*samples)
    s_calc = test.statistic
    a_calc = test.pvalue
    conclusion = 'H0: medians of all groups are equal' if a_calc >= a_level else \
        'H1: at least one population median of one group is different from \
            the population median of at least one other group'
    
    # DataFrame для сводки результатов    
    result = pd.DataFrame({
        'test': ('Kruskal-Wallis test'),
        'p_level': (p_level),
        'a_level': (a_level),
        #'null hypothesis': ('H0: The sample distributions are equal'),
        #'alternative hypothesis': ('H1: The sample distributions are not equal'),
        'a_calc': (a_calc),
        'a_calc >= a_level': (a_calc >= a_level),
        'statistic': (s_calc),
        #'critical_value': ('-'),
        #'statistic <= critical_value': ('-'),
        #'hypothesis check': (''),
        'conclusion (accepted hypothesis)': (conclusion),
        },
        index=[0])
    
    return result



#------------------------------------------------------------------------------
#   Функция Ansari_Bradley_test
#   
#   Тест Ансари-Бредли - непараметрический тест на равенство параметра 
#   масштаба распределений, из которых были взяты две выборки. 
#   H0 утверждает, что отношение масштаба распределения, лежащего в основе x , 
#   к масштабу распределения, лежащего в основе y, равно 1.
#------------------------------------------------------------------------------

def Ansari_Bradley_test(
    data1, data2,                   
    alternative:    str = 'two-sided',
    p_level:        float = 0.95):
    
    """
    Функция реализует тест Ансари-Бредли - непараметрический тест на равенство 
    параметра масштаба распределений, из которых были взяты две выборки. 
    H0 утверждает, что отношение масштаба распределения, лежащего в основе x, 
    к масштабу распределения, лежащего в основе y, равно 1.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ansari.html#scipy-stats-ansari
        
    Args:
        data1, data2:                       
            исходные массивы данных. Должны иметь одинаковый размер.
            
        alternative (str, optional):        
            альтернативная гипотеза {‘two-sided’, ‘less’, ‘greater’}.                                            
            Defaults to ‘two-sided’.
        
        p_level (float, optional):          
            доверительная вероятность. 
            Defaults to 0.95

    Returns:
        result (pd.core.frame.DataFrame):
            результат.
            
    """    
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sps.ansari(
        data1, data2,
        alternative = alternative)
    s_calc = test.statistic    # расчетное значение статистики критерия
    a_calc = test.pvalue    # расчетный уровень значимости
    #hypothesis_check = 'H0' if a_calc >= a_level else 'H1'
    conclusion = 'H0: The scales of the distributions are equal' \
        if a_calc >= a_level else \
            'H1: The scales of the distributions are not equal'
    
    # DataFrame для сводки результатов    
    result = pd.DataFrame({
        'test': ('Ansari-Bradley test'),
        'p_level': (p_level),
        'a_level': (a_level),
        #'null hypothesis': ('H0: The sample distributions are equal'),
        #'alternative hypothesis': ('H1: The sample distributions are not equal'),
        'a_calc': (a_calc),
        'a_calc >= a_level': (a_calc >= a_level),
        'statistic': (s_calc),
        #'critical_value': ('-'),
        #'statistic <= critical_value': ('-'),
        #'hypothesis check': (''),
        'conclusion (accepted hypothesis)': (conclusion),
        },
        index=[0])
    
    return result



#------------------------------------------------------------------------------
#   Функция Mood_test
#   
#   Тест Муда - непараметрический тест нулевой гипотезы о том, что две выборки 
#   взяты из одного и того же распределения с одним и тем же параметром шкалы.
#------------------------------------------------------------------------------

def Mood_test(
    data1, data2,                   
    alternative:    str = 'two-sided',
    p_level:        float = 0.95):
    
    """
    Функция реализует тест Муда - непараметрический тест нулевой гипотезы 
    о том, что две выборки взяты из одного и того же распределения 
    с одним и тем же параметром шкалы..
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mood.html#scipy-stats-mood
        
    Args:
        data1, data2:                       
            исходные массивы данных. Должны иметь одинаковый размер.
            
        alternative (str, optional):        
            альтернативная гипотеза {‘two-sided’, ‘less’, ‘greater’}.                                            
            Defaults to ‘two-sided’.
            
        p_level (float, optional):          
            доверительная вероятность. 
            Defaults to 0.95.
    
    Returns:
        result (pd.core.frame.DataFrame):
            результат.
            
    """    
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sps.mood(
        data1, data2,
        alternative = alternative)
    s_calc = test.statistic    # расчетное значение статистики критерия
    a_calc = test.pvalue    # расчетный уровень значимости
    #hypothesis_check = 'H0' if a_calc >= a_level else 'H1'
    conclusion = 'H0: The samples are drawn from the same distribution with the same scale parameter' \
        if a_calc >= a_level else \
            'H1: samples are not drawn from the same distribution with the same scale parameter'
    
    # DataFrame для сводки результатов    
    result = pd.DataFrame({
        'test': ('Mood test'),
        'p_level': (p_level),
        'a_level': (a_level),
        #'null hypothesis': ('H0: The sample distributions are equal'),
        #'alternative hypothesis': ('H1: The sample distributions are not equal'),
        'a_calc': (a_calc),
        'a_calc >= a_level': (a_calc >= a_level),
        'statistic': (s_calc),
        #'critical_value': ('-'),
        #'statistic <= critical_value': ('-'),
        #'hypothesis check': (''),
        'conclusion (accepted hypothesis)': (conclusion),
        },
        index=[0])
    
    return result



#------------------------------------------------------------------------------
#   Функция Levene_test
#   
#   Критерий Левена
#------------------------------------------------------------------------------

def Levene_test(
    *samples,
    center = 'median',
    p_level: float = 0.95):
    
    """
    Функция реализует тест Левена - проверяет нулевую гипотезу о том, 
    что все входные выборки взяты из совокупностей с равными дисперсиями. 
    Тест Левена является альтернативой тесту Бартлетта
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html#scipy-stats-levene
        
    Args:
        samples:                      
            двумерный массив, содержащий исходные выборки.
            
        center (int, optional):       
            какую функцию данных использовать в тесте 
            {‘mean’, ‘median’, ‘trimmed’}.                                      
            Defaults to ‘median’.
            
        p_level (float, optional):    
            доверительная вероятность. 
            Defaults to 0.95.

    Returns:
        result (pd.core.frame.DataFrame):
            результат.
            
    """ 
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sps.levene(*samples)
    s_calc = test.statistic
    a_calc = test.pvalue
    conclusion = 'H0: The variances of the distributions are equal' \
        if a_calc >= a_level else \
        'H1: The variances of the distributions are not equal'
    
    # DataFrame для сводки результатов    
    result = pd.DataFrame({
        'test': ('Levene test'),
        'p_level': (p_level),
        'a_level': (a_level),
        #'null hypothesis': ('H0: The sample distributions are equal'),
        #'alternative hypothesis': ('H1: The sample distributions are not equal'),
        'a_calc': (a_calc),
        'a_calc >= a_level': (a_calc >= a_level),
        'statistic': (s_calc),
        #'critical_value': ('-'),
        #'statistic <= critical_value': ('-'),
        #'hypothesis check': (''),
        'conclusion (accepted hypothesis)': (conclusion),
        },
        index=[0])
    
    return result



#------------------------------------------------------------------------------
#   Функция Fligner_Killeen_test
#   
#   Критерий Флиннера-Киллина
#------------------------------------------------------------------------------

def Fligner_Killeen_test(
    *samples,
    center      = 'median',
    p_level:    float = 0.95):
    
    """
    Функция реализует тест Флиннера-Киллина - проверяет нулевую гипотезу о том, 
    что все входные выборки взяты из совокупностей с равными дисперсиями. 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fligner.html#scipy-stats-fligner
        
    Args:
        samples:                      
            двумерный массив, содержащий исходные выборки.
            
        center (int, optional):       
            какую функцию данных использовать в тесте 
            {‘mean’, ‘median’, ‘trimmed’}.      
            Defaults to ‘median’.
            
        p_level (float, optional):    
            доверительная вероятность. 
            Defaults to 0.95д
            
    Returns:
        result (pd.core.frame.DataFrame):
            результат.
    """ 
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sps.fligner(*samples)
    s_calc = test.statistic
    a_calc = test.pvalue
    conclusion = 'H0: The variances of the distributions are equal' \
        if a_calc >= a_level else \
        'H1: The variances of the distributions are not equal'
    
    # DataFrame для сводки результатов    
    result = pd.DataFrame({
        'test': ('Fligner-Killeen test'),
        'p_level': (p_level),
        'a_level': (a_level),
        #'null hypothesis': ('H0: The sample distributions are equal'),
        #'alternative hypothesis': ('H1: The sample distributions are not equal'),
        'a_calc': (a_calc),
        'a_calc >= a_level': (a_calc >= a_level),
        'statistic': (s_calc),
        #'critical_value': ('-'),
        #'statistic <= critical_value': ('-'),
        #'hypothesis check': (''),
        'conclusion (accepted hypothesis)': (conclusion),
        },
        index=[0])
    
    return result




#------------------------------------------------------------------------------
#   Функция Abbe_test
#
#   Проверяет гипотезу о наличии сдвига (тренда) средних значений
#   по критерию Аббе
#------------------------------------------------------------------------------

def Abbe_test(X, p_level=0.95):
    a_level = 1 - p_level
    X = np.array(X)
    n = len(X)
        
    # расчетное значение статистики критерия
    if n >= 4:
        Xmean = np.mean(X)
        sum1 = np.sum((X - Xmean)**2)
        sum2 = np.sum([(X[i+1] - X[i])**2 for i in range(n-1)])
        q_calc = 0.5*sum2/sum1
    else:
        q_calc = '-'
        q_table = '-'
        a_calc = '-'
    
    # табличное значение статистики критерия при 4 <= n <= 60
    if n >= 4 and n <= 60:
        Abbe_table_df = pd.read_csv(
            filepath_or_buffer='table/Abbe_test_table.csv',
            sep=';',
            index_col='n')
        p_level_dict = {
            0.95: Abbe_table_df.columns[0],
            0.99: Abbe_table_df.columns[1]}
        f_lin = sci.interpolate.interp1d(Abbe_table_df.index, Abbe_table_df[p_level_dict[p_level]])
        q_table = float(f_lin(n))
    
    #if n >= 20:
    #    a_calc = 1 - sci.stats.norm.cdf(q_calc, loc=1, scale=sqrt((n-2)/(n**2-1)))
    
    # табличное значение статистики критерия при n > 60 (см.Кобзарь, с.517)
    if n > 60:
        u_p = sps.norm.ppf(1-p_level, 0, 1)
        q_table = 1 + u_p / sqrt(n + 0.5*(1 + u_p**2))
        Q_calc = -(1 - q_calc) * sqrt((2*n + 1)/(2 - (1-q_calc)**2))
        Q_table = u_p
        
    # проверка гипотезы
    if n >= 4:
        conclusion = 'independent observations' if q_calc >= q_table else 'dependent observations'    
    else:        
        conclusion = 'count less than 4'
        
    # формируем результат            
    result = pd.DataFrame({
        'n': (n),
        'p_level': (p_level),
        'a_level': (a_level),
        'q_calc': (q_calc),
        'q_table': (q_table if n >= 4 else '-'),
        'q_calc ≥ q_table': (q_calc >= q_table if n >= 4 else '-'),
        'Q_calc (for n > 60)': (Q_calc if n > 60 else '-'),
        'Q_table': (Q_table if n > 60 else '-'),
        'Q_calc ≥ Q_table': (Q_calc >= Q_table if n > 60 else '-'),
        #'a_calc': (a_calc if n > 20 else '-'),
        #'a_calc ≤ a_level': (a_calc <= a_level if n > 20 else '-'),
        'conclusion': (conclusion)
        },
        index=['Abbe test'])
    
    return result



#------------------------------------------------------------------------------
#   Функция Cox_Stuart_test
#   Проверяет гипотезу о случайности значений ряда по критерию Кокса-Стюарта
#------------------------------------------------------------------------------

def Cox_Stuart_test(data, p_level=0.95):
    a_level = 1 - p_level
    data = np.array(data)
    N = len(data)
    
    # функция, выполняющая процедуру расчета теста Кокса-Стюарта
    def calculate_test(X):
        n = len(X)
        # расчетное значение статистики критерия (тренд средних)
        h = lambda i, j: 1 if X[i] > X[j] else 0
        S = np.sum([(n-2*i+1) * h(i-1, (n-i+1)-1) for i in range(1, n//2 + 1)])
        MS = (n**2)/8
        DS = n*(n**2 - 1) / 24
        S_calc = abs(S - MS) / sqrt(DS)
        # табличное значение статистики критерия (тренд средних)
        S_table = sps.norm.ppf((1+p_level)/2, 0, 1)
        # результат
        return S_calc, S_table
    
    # ПРОВЕРКА ГИПОТЕЗЫ О НАЛИЧИИ ТРЕНДА В СРЕДНЕМ
    (S1_calc, S1_table) = calculate_test(data)
    conclusion_mean = 'independent observations' if S1_calc < S1_table else 'dependent observations'    
    
    # ПРОВЕРКА ГИПОТЕЗЫ О НАЛИЧИИ ТРЕНДА В ДИСПЕРСИИ
    # задаем шкалу для объема подвыборок
    k_scale = [
        [48, 2],
        [64, 3],
        [90, 4]]
    # определяем объем подвыборки
    for i, elem in enumerate(k_scale):
        if N < elem[0]:
            K = elem[1]
            break
        else:
            K = 5
    #print(f'N = {N}')
    #print(f'K = {K}')
    # определяем число подвыборок
    R = N//K
    #print(f'R = {R}')
    # формируем подвыборки
    Subsampling = np.zeros((R, K))
    if not R % 2:    # четное число подвыборок
        R_2 = int(R/2)
        for i in range(R_2):
            Subsampling[i] = [data[i*K + j] for j in range(0, K, 1)]
            Subsampling[R - 1 - i] = [data[N - (i*K + j)] for j in range(K, 0, -1)]
    else:    # нечетное число подвыборок
        R_2 = int((R)/2)+1
        for i in range(R_2):
            Subsampling[i] = [data[i*K + j] for j in range(0, K, 1)]
            Subsampling[R - 1 - i] = [data[N - (i*K + j)] for j in range(K, 0, -1)]
    #print(f'Subsampling = \n{Subsampling}\n')
    # проверка гипотезы
    W = [np.amax(Subsampling[i]) - np.amin(Subsampling[i]) for i in range(R)]    # размахи подвыборок
    #print(f'W = {W}')
    (S2_calc, S2_table) = calculate_test(W)
    conclusion_variance = 'independent observations' if S2_calc < S2_table else 'dependent observations'    
    
    # формируем результат            
    result = pd.DataFrame({
        'n': (N),
        'p_level': (p_level),
        'a_level': (a_level),
        'S_calc': (S1_calc, S2_calc),
        'S_table': (S1_table, S2_table),
        'S_calc < S_table': (S1_calc < S1_table, S2_calc < S2_table),
        'conclusion': (conclusion_mean, conclusion_variance)
        },
        index=['Cox_Stuart_test (trend in means)', 'Cox_Stuart_test (trend in variances)'])
    
    return result



#------------------------------------------------------------------------------
#   Функция Foster_Stuart_test
#   Проверяет гипотезу о случайности значений ряда по критерию Фостера-Стюарта
#------------------------------------------------------------------------------

def Foster_Stuart_test(X, p_level=0.95):
    a_level = 1 - p_level
    X = np.array(X)
    n = len(X)
        
    # расчетные значения статистики критерия
    u = l = list()
    Xtemp = np.array(X[0])
    for i in range(1, n):
        Xmax = np.max(Xtemp)
        Xmin = np.min(Xtemp)
        u = np.append(u, 1 if X[i] > Xmax else 0)
        l = np.append(l, 1 if X[i] < Xmin else 0)
        Xtemp = np.append(Xtemp, X[i])
                
    d = np.int64(np.sum(u - l))
    S = np.int64(np.sum(u + l))
        
    # нормализованные расчетные значения статистики критерия
    mean_d = 0
    mean_S = 2*np.sum([1/i for i in range(2, n+1)])
    std_d = sqrt(mean_S)
    std_S = sqrt(mean_S - 4*np.sum([1/i**2 for i in range(2, n+1)]))
    
    '''print(f'mean_d = {mean_d}')
    print(f'std_d = {std_d}')
    print(f'mean_S = {mean_S}')
    print(f'std_S = {std_S}')'''
    
    t_d = (d - mean_d)/std_d
    t_S = (S - mean_S)/std_S
    
    # табличные значения статистики критерия    
    df = n
    t_table = sci.stats.t.ppf((1 + p_level)/2 , df)
    
    # проверка гипотезы
    conclusion_d = 'independent observations' if t_d <= t_table else 'dependent observations'
    conclusion_S = 'independent observations' if t_S <= t_table else 'dependent observations'
    
    # формируем результат            
    result = pd.DataFrame({
        'n': (n),
        'p_level': (p_level),
        'a_level': (a_level),
        'notation': ('d', 'S'),
        'statistic': (d, S),
        'normalized_statistic': (t_d, t_S),
        'crit_value': (t_table),
        'normalized_statistic ≤ crit_value': (t_d <= t_table, t_S <= t_table),
        'conclusion': (conclusion_d, conclusion_S)
        },
        index=['Foster_Stuart_test (trend in means)', 'Foster_Stuart_test (trend in variances)'])
    
    return result

    

#==============================================================================
#               5. ФУНКЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ
#==============================================================================    
    

#------------------------------------------------------------------------------
#   Функция graph_hist_boxplot_probplot_sns
#
#   Визуализация: строит совмещенный график - 
#   гистограмму, коробчатую диаграмму и вероятностный график -  
#   на одном рисунке (Figure)
#------------------------------------------------------------------------------

def graph_hist_boxplot_probplot_sns(
    data,
    data_min                  = None, 
    data_max                  = None,
    graph_inclusion:          str = 'hbp',
    bins_hist:                str = 'auto',
    density_hist:             bool = False,
    type_probplot:            str = 'pp',
    title_figure:             str = None, 
    title_figure_fontsize:    int = 10,
    title_axes:               str = None, 
    title_axes_fontsize:      int = 14,
    data_label:               str = None,
    label_fontsize:           int = 10, 
    tick_fontsize:            int = 8, 
    label_legend_fontsize:    int = 10,
    graph_size:               tuple = None,
    file_name:                str = None):
    
    """
    Визуализация: строит совмещенный график - 
    гистограмму, коробчатую диаграмму и вероятностный график - 
    на одном рисунке (Figure)

    Args:
        data:                                     
            исходный массив данных.
            
        data_min, data_max (optional):            
            мин. и макс. значение рассматриваемой величины для отображения 
            на графиках. 
            Defaults to None.
            
        graph_inclusion (str, optional):          
            параметр, определяющий набор графиков на одном рисунке (Figure).                                                  
            Defaults to 'hbp', где:                                                      
                'h' - hist (гистограмма)                                                      
                'b' - boxplot (коробчатая диаграмма)                                                      
                'p' - probplot (вероятностный график)
                
        bins_hist (str, optional):                
            способ определения числа интервалов гистограммы:
            ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
            # https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges                                                  
            Defaults to 'auto'.
            
        density_hist (bool, optional):            
            вид гистограммы: с относительными/абсолютными частоты (True/False). 
            Defaults to False.
            
        type_probplot (str, optional):            
            вид вероятностного графика ('pp', 'qq'). 
            Defaults to 'pp'.
            
        title_figure (str, optional):             
            заголовок рисунка (Figure). 
            Defaults to None.
            
        title_figure_fontsize (int, optional):    
            размер шрифта заголовка рисунка (Figure). 
            Defaults to 10.
            
        title_axes (str, optional):               
            заголовок области рисования (Axes). 
            Defaults to None.
        
        title_axes_fontsize (int, optional):      
            размер шрифта заголовка области рисования (Axes). 
            Defaults to 14.
        
        data_label (str, optional):              
            подпись оси абсцисс. 
            Defaults to None.
        
        label_fontsize (int, optional):           
            размер шрифта подписи оси абсцисс. 
            Defaults to 10.
        
        tick_fontsize (int, optional):            
            размер шрифта меток оси абсцисс. 
            Defaults to 8.
        
        label_legend_fontsize (int, optional):    
            размер шрифта легенды. 
            Defaults to 10.
        
        graph_size (tuple, optional):             
            размер графика в дюймах. 
            Defaults to None.
            
        file_name (str, optional):                
            имя файла для сохранения на диске. 
            Defaults to None.    
            
    Returns:
        None
            
    """
    
    X = np.array(data)
    Xmin = data_min
    Xmax = data_max
        
    if not(Xmin) and not(Xmax):
        Xmin=min(X)*0.99
        Xmax=max(X)*1.01
        
    # определение формы и размеров области рисования (Axes)
    count_graph = len(graph_inclusion)    # число графиков
    
    ax_rows = count_graph    # размерность области рисования (Axes)
    ax_cols = 1
    
    # создание рисунка (Figure) и области рисования (Axes)
    graph_size_dict = {
        1: (210/INCH, 297/INCH/2),
        2: (210/INCH, 297/INCH/1.5),
        3: (210/INCH, 297/INCH)}
    
    if not(graph_size):
        graph_size = graph_size_dict[count_graph]
    
    fig = plt.figure(figsize=graph_size)
    
    if count_graph == 3:
        ax1 = plt.subplot(3,1,1)
        ax2 = plt.subplot(3,1,2)
        ax3 = plt.subplot(3,1,3)
    elif count_graph == 2:
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
    elif count_graph == 1:
        ax1 = plt.subplot(1,1,1)
       
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    
    # заголовок области рисования (Axes)
    ax1.set_title(title_axes, fontsize = title_axes_fontsize)
    
    # гистограмма (hist)
    if 'h' in graph_inclusion:
        X_mean = X.mean()
        X_std = X.std(ddof = 1)
        # выбор числа интервалов 
        #('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
        bins_hist = bins_hist
        # данные для графика плотности распределения
        nx = 100
        hx = (Xmax - Xmin)/(nx - 1)
        x1 = np.linspace(Xmin, Xmax, nx)
        xnorm1 = sps.norm.pdf(x1, X_mean, X_std)
        kx = len(np.histogram(X, bins=bins_hist, density=density_hist)[0])
        xnorm2 = xnorm1*len(X)*(max(X)-min(X))/kx
        g_kde = sps.gaussian_kde(X)
        g_kde_values = g_kde(x1)
        xnorm3 = g_kde_values*len(X)*(max(X)-min(X))/kx
        # выбор вида гистограммы 
        # (density=True/False - плотность/абс.частота) и параметры по оси OY
        xmax = max(np.histogram(X, bins=bins_hist, density=density_hist)[0])
        ax1.set_ylim(0, xmax*1.4)
        if density_hist:
            label_hist = "empirical density of distribution"
            ax1.set_ylabel('Relative density', fontsize = label_fontsize)
        else:
            label_hist = "empirical frequency"
            ax1.set_ylabel('Absolute frequency', fontsize = label_fontsize)
        # рендеринг графика
        if density_hist:
            ax1.hist(
                X,
                bins=bins_hist,
                density=density_hist,
                histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
                orientation='vertical',
                #color = "#1f77b4",
                color = list(sns.color_palette("husl", 9))[6],
                label=label_hist)
            ax1.plot(
                x1, xnorm1,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'theoretical normal curve')
            sns.kdeplot(
                data = X,
                color = 'magenta',
                label = 'KDE',
                ax=ax1)
        else:
            ax1.hist(
                X,
                bins=bins_hist,
                density=density_hist,
                histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
                orientation='vertical',
                #color = "#1f77b4",
                color = list(sns.color_palette("husl", 9))[6],
                label=label_hist)    
            ax1.plot(
                x1, xnorm2,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'theoretical normal curve')
            ax1.plot(
                x1, xnorm3,
                linestyle = "-",
                color = "magenta",
                #linewidth = 2,
                label = 'KDE')
        ax1.axvline(
            X_mean,
            color='cyan', label = 'mean', linewidth = 2)
        ax1.axvline(
            np.median(X),
            color='orange', label = 'median', linewidth = 2)
        ax1.axvline(stat.mode(X),
            color='green', label = 'mode', linewidth = 2)
        ax1.set_xlim(Xmin, Xmax)
        ax1.legend(fontsize = label_legend_fontsize)
        ax1.tick_params(labelsize = tick_fontsize)
        if (graph_inclusion == 'h') or (graph_inclusion == 'hp'):
            ax1.set_xlabel(data_label, fontsize = label_fontsize)
        ax1.grid()
            
    # коробчатая диаграмма
    if 'b' in graph_inclusion:
        sns.boxplot(
            x=X,
            orient='h',
            width=0.3,
            #color = "#1f77b4",
            color = list(sns.color_palette("husl", 9))[6],
            ax = ax1 if ((graph_inclusion == 'b') or 
                         (graph_inclusion == 'bp')) else ax2)
        if (graph_inclusion == 'b') or (graph_inclusion == 'bp'):
            ax1.set_xlim(Xmin, Xmax)
            ax1.set_xlabel(data_label, fontsize = label_fontsize)
            ax1.tick_params(labelsize = tick_fontsize)
            ax1.yaxis.grid()
        else:
            ax2.set_xlim(Xmin, Xmax)
            ax2.set_xlabel(data_label, fontsize = label_fontsize)
            ax2.tick_params(labelsize = tick_fontsize)
            ax2.yaxis.grid()
    
    # вероятностный график
    gofplot = sm.ProbPlot(
        X,
        dist=sci.stats.distributions.norm,
        fit=True)
    if 'p' in graph_inclusion:
        if type_probplot == 'pp':
            gofplot.ppplot(
                line="45",
                #color = "blue",
                color = list(sns.color_palette("husl", 9))[6],
                ax = ax1 if (graph_inclusion == 'p') 
                    else ax3 if (graph_inclusion == 'hbp') else ax2)
            boxplot_xlabel = 'Theoretical probabilities'
            boxplot_ylabel = 'Sample probabilities'
        elif type_probplot == 'qq':
            gofplot.qqplot(
                line="45",
                #color = "blue",
                color = list(sns.color_palette("husl", 9))[6],
                ax = ax1 if (graph_inclusion == 'p') 
                    else ax3 if (graph_inclusion == 'hbp') else ax2)
            boxplot_xlabel = 'Theoretical quantilies'
            boxplot_ylabel = 'Sample quantilies'
        boxplot_legend = ['data', 'normal distribution']
        if (graph_inclusion == 'p'):
            ax1.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax1.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax1.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax1.tick_params(labelsize = tick_fontsize)
            ax1.grid()
        elif (graph_inclusion == 'hbp'):
            ax3.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax3.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax3.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax3.tick_params(labelsize = tick_fontsize)
            ax3.grid()
        else:
            ax2.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax2.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax2.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax2.tick_params(labelsize = tick_fontsize)
            ax2.grid()
            
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return



#------------------------------------------------------------------------------
#   Функция graph_hist_boxplot_probplot_XY_sns
#   Визуализация: строит совмещенный график - 
#   гистограмму, коробчатую диаграмму и вероятностный график -  
#   на одном рисунке (Figure) для двух переменных (X и Y)
#------------------------------------------------------------------------------

def graph_hist_boxplot_probplot_XY_sns(
    data_X, 
    data_Y,
    data_X_min                = None, 
    data_X_max                = None,
    data_Y_min                = None, 
    data_Y_max                = None,
    graph_inclusion:          str='hbp',
    bins_hist:                str = 'auto',
    density_hist:             bool = False,
    type_probplot:            str = 'pp',
    title_figure:             str = None, 
    title_figure_fontsize:    int = 14,
    x_title:                  str='X', 
    y_title:                  str='Y', 
    title_axes_fontsize:      int = 12,
    data_X_label:             str = None,
    data_Y_label:             str = None,
    label_fontsize:           int = 10, 
    tick_fontsize:            int = 8, 
    label_legend_fontsize:    int = 10,
    graph_size:               tuple = (297/INCH, 210/INCH*1.5),
    file_name:                str = None):
    
    """
    Визуализация: строит совмещенный график - 
    гистограмму, коробчатую диаграмму и вероятностный график - 
    для двух переменных (X и Y) на одном рисунке (Figure)

    Args:
        data_X, data_Y:                                     
            исходные массивы данных (переменные X и Y).
            
        data_X_min, data_X_max (optional):            
            мин. и макс. значение рассматриваемой величины для отображения 
            на графиках (переменная X). 
            Defaults to None.
            
        data_Y_min, data_Y_max (optional):            
            мин. и макс. значение рассматриваемой величины для отображения 
            на графиках (переменная Y). 
            Defaults to None.
            
        graph_inclusion (str, optional):          
            параметр, определяющий набор графиков на одном рисунке (Figure).                                                  
            Defaults to 'hbp', где:                                                      
                'h' - hist (гистограмма)                                                      
                'b' - boxplot (коробчатая диаграмма)                                                      
                'p' - probplot (вероятностный график)
                
        bins_hist (str, optional):                
            способ определения числа интервалов гистограммы:
            ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
            # https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges                                                  
            Defaults to 'auto'.
            
        density_hist (bool, optional):            
            вид гистограммы: с относительными/абсолютными частоты (True/False). 
            Defaults to False.
            
        type_probplot (str, optional):            
            вид вероятностного графика ('pp', 'qq'). 
            Defaults to 'pp'.
            
        title_figure (str, optional):             
            заголовок рисунка (Figure). 
            Defaults to None.
            
        title_figure_fontsize (int, optional):    
            размер шрифта заголовка рисунка (Figure). 
            Defaults to 14.
            
        title_axes (str, optional):               
            заголовок области рисования (Axes). 
            Defaults to None.
        
        title_axes_fontsize (int, optional):      
            размер шрифта заголовка области рисования (Axes). 
            Defaults to 12.
        
        data_X_label, data_Y_label (str, optional):              
            подписи оси абсцисс (переменные X и Y).
            Defaults to None.
        
        label_fontsize (int, optional):           
            размер шрифта подписи оси абсцисс. 
            Defaults to 10.
        
        tick_fontsize (int, optional):            
            размер шрифта меток оси абсцисс. 
            Defaults to 8.
        
        label_legend_fontsize (int, optional):    
            размер шрифта легенды. 
            Defaults to 10.
        
        graph_size (tuple, optional):             
            размер графика в дюймах. 
            Defaults to (210/INCH, 297/INCH).
            
        file_name (str, optional):                
            имя файла для сохранения на диске. 
            Defaults to None.    
            
    Returns:
        None
            
    """    
    
    X = np.array(data_X)
    Xmin = data_X_min
    Xmax = data_X_max
    Y = np.array(data_Y)
    Ymin = data_Y_min
    Ymax = data_Y_max
        
    if not(Xmin) and not(Xmax):
        Xmin=min(X)*0.99
        Xmax=max(X)*1.01
    if not(Ymin) and not(Ymax):
        Ymin=min(Y)*0.99
        Ymax=max(Y)*1.01        
        
    # определение формы и размеров области рисования (Axes)
    count_graph = len(graph_inclusion)    # число графиков
    
    ax_rows = count_graph    # размерность области рисования (Axes)
    ax_cols = 2
    
    # создание рисунка (Figure) и области рисования (Axes)
    graph_size_dict = {
        1: (420/INCH, 297/INCH/2),
        2: (420/INCH, 297/INCH/1.5),
        3: (420/INCH, 297/INCH)}
    
    if not(graph_size):
        graph_size = graph_size_dict[count_graph]
    
    fig = plt.figure(figsize=graph_size)
    
    if count_graph == 3:
        ax1 = plt.subplot(3,2,1)
        ax2 = plt.subplot(3,2,2)
        ax3 = plt.subplot(3,2,3)
        ax4 = plt.subplot(3,2,4)
        ax5 = plt.subplot(3,2,5)
        ax6 = plt.subplot(3,2,6)
    elif count_graph == 2:
        ax1 = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,3)
        ax4 = plt.subplot(2,2,4)
    elif count_graph == 1:
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
       
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    
    # заголовок области рисования (Axes)
    ax1.set_title(x_title, fontsize = title_axes_fontsize)
    ax2.set_title(y_title, fontsize = title_axes_fontsize)
    
    # гистограмма (hist) X
    if 'h' in graph_inclusion:
        X_mean = X.mean()
        X_std = X.std(ddof = 1)
        # выбор числа интервалов 
        # ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
        bins_hist = bins_hist
        # данные для графика плотности распределения
        nx = 100
        hx = (Xmax - Xmin)/(nx - 1)
        x1 = np.linspace(Xmin, Xmax, nx)
        xnorm1 = sps.norm.pdf(x1, X_mean, X_std)
        kx = len(np.histogram(X, bins=bins_hist, density=density_hist)[0])
        xnorm2 = xnorm1*len(X)*(max(X)-min(X))/kx
        g_kde_x = sps.gaussian_kde(X)
        g_kde_x_values = g_kde_x(x1)
        xnorm3 = g_kde_x_values*len(X)*(max(X)-min(X))/kx
        # выбор вида гистограммы 
        # (density=True/False - плотность/абс.частота) и параметры по оси OY
        xmax = max(np.histogram(X, bins=bins_hist, density=density_hist)[0])
        ax1.set_ylim(0, xmax*1.4)
        if density_hist:
            label_hist = "empirical density of distribution"
            ax1.set_ylabel('Relative density', fontsize = label_fontsize)
        else:
            label_hist = "empirical frequency"
            ax1.set_ylabel('Absolute frequency', fontsize = label_fontsize)
        # рендеринг графика
        if density_hist:
            ax1.hist(
                X,
                # выбор числа интервалов ('auto', 'fd', 'doane', '    ', 'stone', 'rice', 'sturges', 'sqrt')
                bins=bins_hist,    
                density=density_hist,
                # 'bar', 'barstacked', 'step', 'stepfilled'
                histtype='bar',    
                # 'vertical', 'horizontal'
                orientation='vertical',   
                #color = "#1f77b4",
                color = list(sns.color_palette("husl", 9))[6],
                label=label_hist)
            ax1.plot(
                x1, xnorm1,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'theoretical normal curve')
            sns.kdeplot(
                data = X,
                color = 'magenta',
                label = 'KDE',
                ax=ax1)
        else:
            ax1.hist(
                X,
                # выбор числа интервалов 
                # ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
                bins=bins_hist,    
                density=density_hist,
                # 'bar', 'barstacked', 'step', 'stepfilled'
                histtype='bar',    
                # 'vertical', 'horizontal'
                orientation='vertical',   
                #color = "#1f77b4",
                color = list(sns.color_palette("husl", 9))[6],
                label=label_hist)    
            ax1.plot(
                x1, xnorm2,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'theoretical normal curve')
            ax1.plot(
                x1, xnorm3,
                linestyle = "-",
                color = "magenta",
                #linewidth = 2,
                label = 'KDE')
        ax1.axvline(
            X_mean,
            color='cyan', label = 'mean', linewidth = 2)
        ax1.axvline(
            np.median(X),
            color='orange', label = 'median', linewidth = 2)
        ax1.axvline(stat.mode(X),
            color='green', label = 'mode', linewidth = 2)
        ax1.set_xlim(Xmin, Xmax)
        ax1.legend(fontsize = label_legend_fontsize)
        ax1.tick_params(labelsize = tick_fontsize)
        if (graph_inclusion == 'h') or (graph_inclusion == 'hp'):
            ax1.set_xlabel(data_X_label, fontsize = label_fontsize)
        ax1.grid()

# гистограмма (hist) Y
    if 'h' in graph_inclusion:
        Y_mean = Y.mean()
        Y_std = Y.std(ddof = 1)
        # выбор числа интервалов 
        #('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
        bins_hist = bins_hist
        # данные для графика плотности распределения
        ny = 100
        hy = (Ymax - Ymin)/(ny - 1)
        y1 = np.linspace(Ymin, Ymax, ny)
        ynorm1 = sps.norm.pdf(y1, Y_mean, Y_std)
        ky = len(np.histogram(Y, bins=bins_hist, density=density_hist)[0])
        ynorm2 = ynorm1*len(Y)*(max(Y)-min(Y))/ky
        g_kde_y = sps.gaussian_kde(Y)
        g_kde_y_values = g_kde_y(y1)
        ynorm3 = g_kde_y_values*len(Y)*(max(Y)-min(Y))/ky
        # выбор вида гистограммы 
        # (density=True/False - плотность/абс.частота) и параметры по оси OY
        ymax = max(np.histogram(Y, bins=bins_hist, density=density_hist)[0])
        ax2.set_ylim(0, ymax*1.4)
        if density_hist:
            label_hist = "empirical density of distribution"
            ax2.set_ylabel('Relative density', fontsize = label_fontsize)
        else:
            label_hist = "empirical frequency"
            ax2.set_ylabel('Absolute frequency', fontsize = label_fontsize)
        # рендеринг графика
        if density_hist:
            ax2.hist(
                Y,
                bins=bins_hist,
                density=density_hist,
                histtype='bar',
                orientation='vertical',
                #color = "#1f77b4",
                color = list(sns.color_palette("husl", 9))[6],
                label=label_hist)
            ax2.plot(
                y1, ynorm1,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'theoretical normal curve')
            sns.kdeplot(
                data = Y,
                color = 'magenta',
                label = 'KDE',
                ax=ax2)
        else:
            ax2.hist(
                Y,
                bins=bins_hist,
                density=density_hist,
                histtype='bar',
                orientation='vertical',
                #color = "#1f77b4",
                color = list(sns.color_palette("husl", 9))[6],
                label=label_hist)    
            ax2.plot(
                y1, ynorm2,
                linestyle = "-",
                color = "r",
                linewidth = 2,
                label = 'theoretical normal curve')
            ax2.plot(
                y1, ynorm3,
                linestyle = "-",
                color = "magenta",
                #linewidth = 2,
                label = 'KDE')
        ax2.axvline(
            Y_mean,
            color='cyan', label = 'mean', linewidth = 2)
        ax2.axvline(
            np.median(Y),
            color='orange', label = 'median', linewidth = 2)
        ax2.axvline(stat.mode(Y),
            color='green', label = 'mode', linewidth = 2)
        ax2.set_xlim(Ymin, Ymax)
        ax2.legend(fontsize = label_legend_fontsize)
        ax2.tick_params(labelsize = tick_fontsize)
        if (graph_inclusion == 'h') or (graph_inclusion == 'hp'):
            ax2.set_xlabel(data_Y_label, fontsize = label_fontsize)
        ax2.grid()
        
    # коробчатая диаграмма X
    if 'b' in graph_inclusion:
        sns.boxplot(
            x=X,
            orient='h',
            width=0.3,
            ax = ax1 if (graph_inclusion == 'b') or (graph_inclusion == 'bp') else ax3)
        if (graph_inclusion == 'b') or (graph_inclusion == 'bp'):
            ax1.set_xlim(Xmin, Xmax)
            ax1.set_xlabel(data_X_label, fontsize = label_fontsize)
            ax1.tick_params(labelsize = tick_fontsize)
            ax1.set_xlabel(data_X_label, fontsize = label_fontsize)
            ax1.xaxis.grid()
        else:
            ax3.set_xlim(Xmin, Xmax)
            ax3.set_xlabel(data_X_label, fontsize = label_fontsize)
            ax3.tick_params(labelsize = tick_fontsize)
            ax3.xaxis.grid()
            
    # коробчатая диаграмма Y
    if 'b' in graph_inclusion:
        sns.boxplot(
            x=Y,
            orient='h',
            width=0.3,
            ax = ax2 if ((graph_inclusion == 'b') or 
                         (graph_inclusion == 'bp')) else ax4)
        if (graph_inclusion == 'b') or (graph_inclusion == 'bp'):
            ax2.set_xlim(Ymin, Ymax)
            ax2.set_xlabel(data_Y_label, fontsize = label_fontsize)
            ax2.tick_params(labelsize = tick_fontsize)
            ax2.set_xlabel(data_Y_label, fontsize = label_fontsize)
            ax2.xaxis.grid()
        else:
            ax4.set_xlim(Ymin, Ymax)
            ax4.set_xlabel(data_Y_label, fontsize = label_fontsize)
            ax4.tick_params(labelsize = tick_fontsize)     
            ax4.xaxis.grid()
    
    # вероятностный график X
    gofplot = sm.ProbPlot(
        X,
        dist=sci.stats.distributions.norm,
        fit=True)
    if 'p' in graph_inclusion:
        if type_probplot == 'pp':
            gofplot.ppplot(
                line="45",
                ax = ax1 if (graph_inclusion == 'p') else \
                    ax5 if (graph_inclusion == 'hbp') else ax3)
            boxplot_xlabel = 'Theoretical probabilities'
            boxplot_ylabel = 'Sample probabilities'
        elif type_probplot == 'qq':
            gofplot.qqplot(
                line="45",
                ax = ax1 if (graph_inclusion == 'p') else \
                    ax5 if (graph_inclusion == 'hbp') else ax3)
            boxplot_xlabel = 'Theoretical quantilies'
            boxplot_ylabel = 'Sample quantilies'
        boxplot_legend = ['empirical data', 'normal distribution']
        if (graph_inclusion == 'p'):
            ax1.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax1.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax1.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax1.tick_params(labelsize = tick_fontsize)
            ax1.grid()
        elif (graph_inclusion == 'hbp'):
            ax5.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax5.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax5.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax5.tick_params(labelsize = tick_fontsize)
            ax5.grid()
        else:
            ax3.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax3.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax3.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax3.tick_params(labelsize = tick_fontsize)
            ax3.grid()
            
    # вероятностный график Y
    gofplot = sm.ProbPlot(
        Y,
        dist=sci.stats.distributions.norm,
        fit=True)
    if 'p' in graph_inclusion:
        if type_probplot == 'pp':
            gofplot.ppplot(
                line="45",
                ax = ax2 if (graph_inclusion == 'p') else \
                    ax6 if (graph_inclusion == 'hbp') else ax4)
        elif type_probplot == 'qq':
            gofplot.qqplot(
                line="45",
                ax = ax2 if (graph_inclusion == 'p') else \
                    ax6 if (graph_inclusion == 'hbp') else ax4)
        boxplot_legend = ['empirical data', 'normal distribution']
        if (graph_inclusion == 'p'):
            ax2.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax2.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax2.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax2.tick_params(labelsize = tick_fontsize)
            ax2.grid()
        elif (graph_inclusion == 'hbp'):
            ax6.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax6.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax6.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax6.tick_params(labelsize = tick_fontsize)
            ax6.grid()
        else:
            ax4.set_xlabel(boxplot_xlabel, fontsize = label_fontsize)
            ax4.set_ylabel(boxplot_ylabel, fontsize = label_fontsize)
            ax4.legend(boxplot_legend, fontsize = label_legend_fontsize)        
            ax4.tick_params(labelsize = tick_fontsize) 
            ax4.grid()
            
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return    



#------------------------------------------------------------------------------
#   Функция graph_ecdf_cdf_mpl
#
#   Визуализация: строит совмещенный график - 
#   эмпирическую (ecdf) и теоретическую нормальную (cdf) функцию распределения.
#------------------------------------------------------------------------------

def graph_ecdf_cdf_mpl(
    data,
    data_min                  = None, 
    data_max                  = None,
    title_figure:             str = None, 
    title_figure_fontsize:    int = 10,
    title_axes:               str = None, 
    title_axes_fontsize:      int = 14,
    data_label:               str = None,
    label_fontsize:           int = 10, 
    tick_fontsize:            int = 8, 
    legend_fontsize:          int = 10,
    linewidth_ecdf:           int = 2,
    linewidth_cdf:            int = 2,
    graph_size:               tuple = (210/INCH, 297/INCH/2),
    file_name:                str = None):
    
    """
    Визуализация: строит совмещенный график - 
    эмпирическую (ecdf) и теоретическую нормальную (cdf) функцию распределения.

    Args:
        data:                                     
            исходный массив данных.
        
        data__min, data__max (optional):          
            мин. и макс. значение рассматриваемой величины для отображения 
            на графиках. 
            Defaults to None.
            
        title_figure (str, optional):             
            заголовок рисунка (Figure). 
            Defaults to None.
        
        title_figure_fontsize (int, optional):    
            размер шрифта заголовка рисунка (Figure). 
            Defaults to 10.            
            
        title_axes (str, optional):               
            заголовок области рисования (Axes). 
            Defaults to None.
            
        title_axes_fontsize (int, optional):      
            размер шрифта заголовка области рисования (Axes). 
            Defaults to 14.
            
        data_label (str, optional):              
            подпись оси абсцисс. 
            Defaults to None.
            
        label_fontsize (int, optional):           
            размер шрифта подписи оси абсцисс. 
            Defaults to 10.
            
        tick_fontsize (int, optional):            
            размер шрифта меток оси абсцисс. 
            Defaults to 8.
            
        legend_fontsize (int, optional):    
            размер шрифта легенды. Defaults to 10.
            
        linewidth_ecdf (int, optional):           
            толщина линии графика эмпирической функции распределения. 
            Defaults to 2.
            
        linewidth_cdf (int, optional):            
            толщина линии графика теоретической функции распределения. 
            Defaults to 2.
            
        graph_size (tuple, optional):             
            размер графика в дюймах. 
            Defaults to (210/INCH, 297/INCH/2), 
                где константа INCH = 25.4 мм/дюйм
                
        file_name (str, optional):                
            имя файла для сохранения на диске. 
            Defaults to None.
            
    """
    
    data = np.array(data)
    
    if not(data_min) and not(data_max):
        data_min = min(data_)*0.99
        data_max = max(data_)*1.01
    
    # создание рисунка (Figure) и области рисования (Axes)
    fig = plt.figure(figsize=graph_size)
    ax = plt.subplot(2,1,1)    # для гистограммы
        
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    
    # заголовок области рисования (Axes)
    ax.set_title(title_axes, fontsize = title_axes_fontsize)
    
    # эмпирическая функция распределения
    from statsmodels.distributions.empirical_distribution import ECDF
    x1 = np.array(data)
    ecdf = ECDF(x1)
    x1.sort()
    y1 = ecdf(x1)
        
    # теоретическая функция распределения
    hx = 0.1; nx = floor((data_max - data_min)/hx)+1
    x2 = np.linspace(data_min, data_max, nx)
    y2 = sps.norm.cdf(x2, data.mean(), data.std(ddof = 1))
    
    # рендеринг эмпирической функции распределения
    ax.step(x1, y1,
            where='post',
            linewidth = linewidth_ecdf,
            label = 'empirical distribution function')
    
    # рендеринг теоретической функции распределения
    ax.plot(
       x2, y2,
       linestyle = "-",
       color = "r",
       linewidth = linewidth_cdf,
       label = 'theoretical normal distribution function')
    
    ax.set_xlim(data_min, data_max)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(data_label, fontsize = label_fontsize)
    
    ax.grid(True)
    ax.legend(fontsize = legend_fontsize)
    ax.tick_params(labelsize = tick_fontsize)
    
    # отображение графика на экране и сохранение в файл
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return    

    

#------------------------------------------------------------------------------
#   Функция graph_lorenz_curve
#   
#   Визуализация: строит график кривой Лоренца
#------------------------------------------------------------------------------

def graph_lorenz_curve(
    data,
    title_figure:             str = None, 
    title_figure_fontsize:    int = 14,
    title_axes:               str = None, 
    title_axes_fontsize:      int = 12,
    x_label:                  str = None, 
    y_label:                  str = None,
    label_fontsize:           int = 10, 
    tick_fontsize:            int = 8,
    label_legend_fontsize:    int = 10,
    label_text_fontsize:      int = 10,
    graph_size:               tuple = (210/INCH, 297/INCH/2),
    file_name:                str = None):
    
    """
    Визуализация: строит график кривой Лоренца.

    Args:
        data:                                     
            исходный массив данных.
            
        title_figure (str, optional):             
            заголовок рисунка (Figure). 
            Defaults to None.
            
        title_figure_fontsize (int, optional):    
            размер шрифта заголовка рисунка (Figure). 
            Defaults to 14.
            
        title_axes (str, optional):               
            заголовок области рисования (Axes). 
            Defaults to None.
        
        title_axes_fontsize (int, optional):      
            размер шрифта заголовка области рисования (Axes). 
            Defaults to 12.
            
        x_label, y_label (str, optional):         
            подписи осей абсцисс и ординат. 
            Defaults to None.
            
        label_fontsize (int, optional):           
            размер шрифта подписей осей. 
            Defaults to 10.
            
        tick_fontsize (int, optional):            
            размер шрифта меток осей. 
            Defaults to 8.
            
        label_legend_fontsize (int, optional):    
            размер шрифта легенды. 
            Defaults to 8.
            
        label_text_fontsize (int, optional):      
            размер шрифта текста на поле. 
            Defaults to 10.
            
        graph_size (tuple, optional):             
            размер графика в дюймах. 
            Defaults to (210/INCH, 297/INCH/2), 
                где константа INCH = 25.4 мм/дюйм.
                
        file_name (str, optional):                
            имя файла для сохранения на диске. 
            Defaults to None.   

    Returns:
        None
        
    """
    
    # создание рисунка (Figure) и области рисования (Axes)
    # в случае простого графика области Figure и Axes совпадают
    fig, ax = plt.subplots(figsize = graph_size)
    
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)

    # заголовок области рисования (Axes)
    ax.set_title(title_axes, fontsize = title_axes_fontsize)
    
    # исходные данные
    #--------------------
    
    #№ данные для графика кривой Лоренца
    data = np.array(sorted(data))
        
    Y_cum = data.cumsum() / data.sum()
    Y_cum = np.insert(Y_cum, 0, 0)
        
    X_cum = np.arange(Y_cum.size) / (Y_cum.size-1)
        
    # расчет коэффициента Джини
    G_list = list((X_cum[i] - X_cum[i-1]) * (Y_cum[i] + Y_cum[i-1]) 
                  for i in range(1,len(Y_cum)))
    G = abs(1 - sum(G_list))
        
    # рендеринг диаграммы
    #--------------------
    
    ## кривая Лоренца (line plot of Lorenz curve)
    ax.plot(
        X_cum, Y_cum,
        color = 'blue',
        label='кривая Лоренца')
    
    ## линия абсолютного равенства (line plot of equality)
    ax.plot(
        [0,1], [0,1],
        color = 'black',
        label='линия абсолютного равенства')
    
    ## область неравенства
    ax.fill_between(
        x=np.insert(X_cum,0,0),
        y1=np.insert(X_cum,0,0),
        y2=np.insert(Y_cum,0,0),
        color='lightblue',
        label='область неравенства')
    
    ## значение коэффициента Джини
    plt.text(0.1,0.7,'Коэффициент Джини G = {:0.3f}'.format(G), \
             fontsize = label_text_fontsize)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid()
    ax.legend(fontsize = label_legend_fontsize)
    ax.set_xlabel(x_label, fontsize = label_fontsize)
    ax.set_ylabel(y_label, fontsize = label_fontsize)
    ax.tick_params(labelsize = tick_fontsize)
           
    # отображение графика на экране и сохранение в файл
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return



#------------------------------------------------------------------------------
#   Функция graph_df_heatmap
#   
#   Визуализация матрицы корреляционных связей между признаками датасета 
#   (числовыми и категориальными) с помощью тепловой карты (heatmap).
#------------------------------------------------------------------------------

def graph_df_heatmap(
    data_df_in:                  pd.core.frame.DataFrame,
    data_df_clone:               pd.core.frame.DataFrame = None,
    cols_to_exclude:             list = None,
    column_to_aggregate:         str = None,
    corr_method_qualitative:     str = 'spearman',    
    corr_method_quantitative:    str = 'cramer',      
    DecPlace:                    int = 3,
    title_figure:                str = None, 
    title_figure_fontsize:       int = 12,
    title_axes:                  str = None, 
    title_axes_fontsize:         int = 16,
    label_fontsize:              int = 13, 
    tick_fontsize:               int = 10, 
    label_legend_fontsize:       int = 10,
    colormap                     = 'Reds',
    lower_half:                  bool = False,
    corr_table_output:           bool = True,
    graph_size:                  tuple = (297/INCH/2, 210/INCH/2),
    file_name:                   str = None):
    
    """
    Визуализация матрицы корреляционных связей между признаками датасета 
    (числовыми и категориальными) с помощью тепловой карты (heatmap).

    Args:
        data_df_in (pd.core.frame.DataFrame):              
            исходный датасет.
            
        data_df_clone (pd.core.frame.DataFrame, optional):    
            датасет-клон, в котором числовые признаки заменены на 
            категориальные.
            Defaults to None.
                                                                  
        cols_to_exclude (list, optional):                       
            список-исключение признаков, которые не участвуют в формировании
            корреляционной матрицы.                                                                                
            
            Defaults to None.
        
        column_to_aggregate (str, optional):                        
            признак датасета, который используется для агрегирования 
            числовых данных.                                                                                
            Defaults to None.
            
        corr_method_qualitative (str, optional):                        
            показатель для оценки тесноты связи между числовыми признаками:                                                                                
            {'spearman', 'kendall'}                                                                                
            Defaults to 'spearman'.
            
        corr_method_quantitative (str, optional):                        
            показатель для оценки тесноты связи между категориальными 
            признаками:                                                                                
            {'cramer', 'tschuprow', 'pearson'}                                                                                
            Defaults to 'cramer'.
        
        DecPlace (int, optional):        
            точность (число знаков дробной части) при выводе коэффициентов.
            Defaults to 3.
                
        title_figure (str, optional):                 
            заголовок рисунка (Figure). 
            Defaults to None.
            
        title_figure_fontsize (int, optional):       
            размер шрифта заголовка рисунка (Figure). 
            Defaults to 12.
            
        title_axes (str, optional):               
            заголовок области рисования (Axes). 
            Defaults to None.
        
        title_axes_fontsize (int, optional):      
            размер шрифта заголовка области рисования (Axes). 
            Defaults to 16.
            
        label_fontsize (int, optional):              
            размер шрифта подписей осей.
            Defaults to 13.
            
        tick_fontsize (int, optional):               
            размер шрифта меток. 
            Defaults to 10.
            
        label_legend_fontsize (int, optional):       
            размер шрифта легенды. 
            Defaults to 10.
            
        colormap (optional):                          
            цветовая палитра.
            Defaults to 'Reds'.         
                    
        lower_half (bool, optional):
            логический параметр: True/False - показывать/не показывать нижнюю 
            половину корреляционной марицы.
            Defaults to False.
            
        corr_table_output (bool, optional):
            логический параметр: True/False - выводить/не выводить 
            корреляционную марицу.
            Defaults to True. 
            
        graph_size (tuple, optional):
            размер графика в дюймах. 
            Defaults to (420/INCH*1.4, 297/INCH), 
                где константа INCH = 25.4 мм/дюйм.
                
        file_name (str, optional):                
            имя файла для сохранения на диске. 
            Defaults to None.
    
    Returns:
        result (pd.core.frame.DataFrame, optional):
            результат (корреляционная матрица).
            
    """
    
        # особенность аргумента по умолчанию, который является списком (list)
    cols_to_exclude = cols_to_exclude or []
    
    # формируем рабочий датасет - исключаем столбцы
    data_df_exclude = data_df_in.drop(columns = cols_to_exclude)
        
    # типы столбцов в датасете
    data_df_exclude_dtypes_series = data_df_exclude.dtypes
        
    # формируем копию датасета, в котором количественные признаки 
    # трансформированы в качественные
    data_df = data_df_exclude.copy()
            
    # формируем корреляционную матрицу
    variable_list = list(data_df.columns)
    variable_list.remove(column_to_aggregate)
        
    corr_matrix = np.eye(len(variable_list))
    corr_matrix_df = pd.DataFrame(corr_matrix, index = variable_list, \
                                  columns = variable_list)
       
    # формируем словарь условных обозначений для коэффициентов
    coef_name_dict = {
        'spearman':  chr(961),
        'kendall':   chr(964),
        'pearson':   'C',
        'cramer':    'V',
        'tschuprow': 'T'}
    
    # формируем матрицу аннотаций
    annot_df = corr_matrix_df.copy()
    
    for i, elem_i in enumerate(variable_list):
        for j, elem_j in enumerate(variable_list):
            if j>i:
                type_elem_i = data_df_exclude_dtypes_series[elem_i]
                type_elem_j = data_df_exclude_dtypes_series[elem_j]
                #print(f'{elem_i}, {elem_j}: {type_elem_i} {type_elem_j}')
                if (type_elem_i == 'category' and type_elem_j == 'category'):
                    temp_df = data_df.pivot_table(
                        values = column_to_aggregate,
                        index = elem_i,
                        columns = elem_j,
                        aggfunc = 'count',
                        fill_value = 0)
                    corr_matrix_df.loc[elem_i, elem_j] = \
                        sci.stats.contingency.association(
                        temp_df, 
                        method = corr_method_quantitative, 
                        correction='False')
                    annot_df.loc[elem_i, elem_j] = \
                        f'{coef_name_dict[corr_method_quantitative]} = ' + \
                            str(round(corr_matrix_df.loc[elem_i, elem_j], \
                                      DecPlace))
                    if lower_half:
                        corr_matrix_df.loc[elem_j, elem_i] = \
                            corr_matrix_df.loc[elem_i, elem_j]
                        annot_df.loc[elem_j, elem_i] = \
                            annot_df.loc[elem_i, elem_j]
                    else:
                        corr_matrix_df.loc[elem_j, elem_i] = 0
                        annot_df.loc[elem_j, elem_i] = ''
                
                elif ((type_elem_i != 'category' and type_elem_i != 'str') \
                      and (type_elem_j != 'category' and type_elem_j != 'str')):
                    temp_res = \
                        sci.stats.spearmanr(data_df[elem_i], data_df[elem_j]) \
                            if corr_method_qualitative == 'spearman' \
                                else sci.stats.kendalltau(data_df[elem_i], \
                                    data_df[elem_j]) \
                                        if corr_method_qualitative == \
                                            'kendall' else None
                    corr_matrix_df.loc[elem_i, elem_j] = abs(temp_res.statistic)
                    annot_df.loc[elem_i, elem_j] = \
                        f'{coef_name_dict[corr_method_qualitative]} = ' + \
                            str(round(corr_matrix_df.loc[elem_i, elem_j], \
                                      DecPlace))
                    if lower_half:
                        corr_matrix_df.loc[elem_j, elem_i] = \
                            corr_matrix_df.loc[elem_i, elem_j]                    
                        annot_df.loc[elem_j, elem_i] = \
                            annot_df.loc[elem_i, elem_j]
                    else:
                        corr_matrix_df.loc[elem_j, elem_i] = 0
                        annot_df.loc[elem_j, elem_i] = ''
                        
                    
                else:
                    if data_df_clone is not None:
                        elem_list = list(data_df_clone.columns)
                        if (elem_i in elem_list) and (elem_j in elem_list):
                            temp_df = data_df_clone.pivot_table(
                                values = column_to_aggregate,
                                index = elem_i,
                                columns = elem_j,
                                aggfunc = 'count',
                                fill_value = 0)
                            corr_matrix_df.loc[elem_i, elem_j] = \
                                sci.stats.contingency.association(
                                temp_df, 
                                method = corr_method_quantitative, 
                                correction='False')
                            annot_df.loc[elem_i, elem_j] = \
                                f'{coef_name_dict[corr_method_quantitative]} = '\
                                    + str(round(corr_matrix_df.loc[elem_i, \
                                        elem_j], DecPlace))
                            if lower_half:
                                corr_matrix_df.loc[elem_j, elem_i] = \
                                    corr_matrix_df.loc[elem_i, elem_j]
                                annot_df.loc[elem_j, elem_i] = \
                                    annot_df.loc[elem_i, elem_j]
                            else:
                                corr_matrix_df.loc[elem_j, elem_i] = 0
                                annot_df.loc[elem_j, elem_i] = ''
                                
                        else:
                            annot_df.loc[elem_i, elem_j] = 'no col in df clone'
                            #annot_df.loc[elem_j, elem_i] = annot_df.loc[elem_i, elem_j]                   
                    else:
                        annot_df.loc[elem_i, elem_j] = 'no df clone'
                        #annot_df.loc[elem_j, elem_i] = annot_df.loc[elem_i, elem_j]                   
        
    # построение графика
    fig, axes = plt.subplots(figsize=(graph_size))
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = 16)
    sns.heatmap(
        corr_matrix_df,
        vmin = 0, vmax = 1,
        cbar = True,
        #center = True,
        #annot = True,
        annot = annot_df,
        cmap = colormap,
        #fmt = '.4f',
        fmt = '',
        xticklabels = variable_list,
        yticklabels = variable_list)
    plt.show()      
    
    # вывод графика
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    # вывод результата
    if corr_table_output:
        result = corr_matrix_df
        return result
    else:
        return 



#------------------------------------------------------------------------------
#   Функция transformation_numeric_to_category
#
#   Принимает в качестве аргумента числовое значение и преобразует его 
#   в категориальный признак.
#------------------------------------------------------------------------------  

def transformation_numeric_to_category(
    numeric_in,
    transformation_dict:          dict,
    top_bound_less_than_or_equal: bool = True):
    
    """
    Принимает в качестве аргумента числовое значение и преобразует его 
    в значение категориального признака в соответствии с правилами 
    преобразования, закодированными в специальном словаре для преоразования 
    данных.

    Args:
        numeric_in:     
            числовое значение.
            
        transformation_dict (dict):    
            словарь для преобразования данных. 
                        
        top_bound_less_than_or_equal (bool, optional):    
            логический параметр, определяющий правило выбора категории 
            в словаре для преобразования данных в зависимости от числового
            значения: по умолчанию True, то есть применеятся правило "меньше 
            или равно". 
            Defaults to True.
        
    Returns:    
        data_df (pd.core.frame.DataFrame):
            преобразованный DataFrame            
            
    """
    
    transformation_scale = list(transformation_dict.values())
    if top_bound_less_than_or_equal:
        #if not numeric_in or isnan(numeric_in):
        if isnan(numeric_in):
            result = None
        else:
            for i, elem in enumerate(transformation_scale):
                if abs(numeric_in) <= elem:
                    result = list(transformation_dict.keys())[i]
                    break
    else: 
        #if not numeric_in or isnan(numeric_in):
        if isnan(numeric_in):
            result = None
        else:
            for i, elem in enumerate(transformation_scale):
                if abs(numeric_in) < elem:
                    result = list(transformation_dict.keys())[i]
                    break
    
    return result        



#------------------------------------------------------------------------------
#   Функция graph_df_heatmap
#   
#   Визуализация матрицы корреляционных связей между признаками датасета 
#   (числовыми и категориальными) с помощью тепловой карты (heatmap).
#------------------------------------------------------------------------------

def graph_df_heatmap(
    data_df_in:                  pd.core.frame.DataFrame,
    data_df_clone:               pd.core.frame.DataFrame = None,
    cols_to_exclude:             list = None,
    column_to_aggregate:         str = None,
    corr_method_qualitative:     str = 'spearman',    
    corr_method_quantitative:    str = 'cramer',      
    DecPlace:                    int = 3,
    title_figure:                str = None, 
    title_figure_fontsize:       int = 10,
    title_axes:                  str = None, 
    title_axes_fontsize:         int = 12,
    label_fontsize:              int = 10, 
    tick_fontsize:               int = 8, 
    label_legend_fontsize:       int = 10,
    annot_fontsize:              int = 8,
    colorbar_fontsize:           int = 8,
    colormap                     = 'Reds',
    lower_half:                  bool = False,
    corr_table_output:           bool = True,
    graph_size:                  tuple = (420/INCH/1.5, 297/INCH/1.5),
    file_name:                   str = None):
    
    """
    Визуализация матрицы корреляционных связей между признаками датасета 
    (числовыми и категориальными) с помощью тепловой карты (heatmap).

    Args:
        data_df_in (pd.core.frame.DataFrame):              
            исходный датасет.
            
        data_df_clone (pd.core.frame.DataFrame, optional):    
            датасет-клон, в котором числовые признаки заменены на 
            категориальные.
            Defaults to None.
                                                                  
        cols_to_exclude (list, optional):                       
            список-исключение признаков, которые не участвуют в формировании
            корреляционной матрицы.                                                                                
            
            Defaults to None.
        
        column_to_aggregate (str, optional):                        
            признак датасета, который используется для агрегирования 
            числовых данных.                                                                                
            Defaults to None.
            
        corr_method_qualitative (str, optional):                        
            показатель для оценки тесноты связи между числовыми признаками:                                                                                
            {'spearman', 'kendall'}                                                                                
            Defaults to 'spearman'.
            
        corr_method_quantitative (str, optional):                        
            показатель для оценки тесноты связи между категориальными 
            признаками:                                                                                
            {'cramer', 'tschuprow', 'pearson'}                                                                                
            Defaults to 'cramer'.
        
        DecPlace (int, optional):        
            точность (число знаков дробной части) при выводе коэффициентов.
            Defaults to 3.
                
        title_figure (str, optional):                 
            заголовок рисунка (Figure). 
            Defaults to None.
            
        title_figure_fontsize (int, optional):       
            размер шрифта заголовка рисунка (Figure). 
            Defaults to 10.
            
        title_axes (str, optional):               
            заголовок области рисования (Axes). 
            Defaults to None.
        
        title_axes_fontsize (int, optional):      
            размер шрифта заголовка области рисования (Axes). 
            Defaults to 12.
            
        label_fontsize (int, optional):              
            размер шрифта подписей осей.
            Defaults to 10.
            
        tick_fontsize (int, optional):               
            размер шрифта меток. 
            Defaults to 8.
            
        label_legend_fontsize (int, optional):       
            размер шрифта легенды. 
            Defaults to 10.
        
        annot_fontsize (int, optional):       
            размер шрифта подписей в центре плиток графика. 
            Defaults to 8.
        
        colorbar_fontsize (int, optional):       
            размер шрифта подписей цветовой полосы. 
            Defaults to 8.
            
        colormap (optional):                          
            цветовая палитра.
            Defaults to 'Reds'.         
                    
        lower_half (bool, optional):
            логический параметр: True/False - показывать/не показывать нижнюю 
            половину корреляционной марицы.
            Defaults to False.
            
        corr_table_output (bool, optional):
            логический параметр: True/False - выводить/не выводить 
            корреляционную марицу.
            Defaults to True. 
            
        graph_size (tuple, optional):
            размер графика в дюймах. 
            Defaults to (420/INCH*1.4, 297/INCH), 
                где константа INCH = 25.4 мм/дюйм.
                
        file_name (str, optional):                
            имя файла для сохранения на диске. 
            Defaults to None.
    
    Returns:
        result (pd.core.frame.DataFrame, optional):
            результат (корреляционная матрица).
            
    """
    
        # особенность аргумента по умолчанию, который является списком (list)
    cols_to_exclude = cols_to_exclude or []
    
    # формируем рабочий датасет - исключаем столбцы
    data_df_exclude = data_df_in.drop(columns = cols_to_exclude)
        
    # типы столбцов в датасете
    data_df_exclude_dtypes_series = data_df_exclude.dtypes
        
    # формируем копию датасета, в котором количественные признаки 
    # трансформированы в качественные
    data_df = data_df_exclude.copy()
            
    # формируем корреляционную матрицу
    variable_list = list(data_df.columns)
    variable_list.remove(column_to_aggregate)
        
    corr_matrix = np.eye(len(variable_list))
    corr_matrix_df = pd.DataFrame(corr_matrix, index = variable_list, \
                                  columns = variable_list)
       
    # формируем словарь условных обозначений для коэффициентов
    coef_name_dict = {
        'spearman':  chr(961),
        'kendall':   chr(964),
        'pearson':   'C',
        'cramer':    'V',
        'tschuprow': 'T'}
    
    # формируем матрицу аннотаций
    annot_df = corr_matrix_df.copy()
    
    for i, elem_i in enumerate(variable_list):
        for j, elem_j in enumerate(variable_list):
            if j>i:
                type_elem_i = data_df_exclude_dtypes_series[elem_i]
                type_elem_j = data_df_exclude_dtypes_series[elem_j]
                #print(f'{elem_i}, {elem_j}: {type_elem_i} {type_elem_j}')
                if (type_elem_i == 'category' and type_elem_j == 'category'):
                    temp_df = data_df.pivot_table(
                        values = column_to_aggregate,
                        index = elem_i,
                        columns = elem_j,
                        aggfunc = 'count',
                        fill_value = 0)
                    corr_matrix_df.loc[elem_i, elem_j] = \
                        sci.stats.contingency.association(
                        temp_df, 
                        method = corr_method_quantitative, 
                        correction='False')
                    annot_df.loc[elem_i, elem_j] = \
                        f'{coef_name_dict[corr_method_quantitative]} = ' + \
                            str(round(corr_matrix_df.loc[elem_i, elem_j], \
                                      DecPlace))
                    if lower_half:
                        corr_matrix_df.loc[elem_j, elem_i] = \
                            corr_matrix_df.loc[elem_i, elem_j]
                        annot_df.loc[elem_j, elem_i] = \
                            annot_df.loc[elem_i, elem_j]
                    else:
                        corr_matrix_df.loc[elem_j, elem_i] = 0
                        annot_df.loc[elem_j, elem_i] = ''
                
                elif ((type_elem_i != 'category' and type_elem_i != 'str') \
                      and (type_elem_j != 'category' and type_elem_j != 'str')):
                    temp_res = \
                        sci.stats.spearmanr(data_df[elem_i], data_df[elem_j]) \
                            if corr_method_qualitative == 'spearman' \
                                else sci.stats.kendalltau(data_df[elem_i], \
                                    data_df[elem_j]) \
                                        if corr_method_qualitative == \
                                            'kendall' else None
                    corr_matrix_df.loc[elem_i, elem_j] = abs(temp_res.statistic)
                    annot_df.loc[elem_i, elem_j] = \
                        f'{coef_name_dict[corr_method_qualitative]} = ' + \
                            str(round(corr_matrix_df.loc[elem_i, elem_j], \
                                      DecPlace))
                    if lower_half:
                        corr_matrix_df.loc[elem_j, elem_i] = \
                            corr_matrix_df.loc[elem_i, elem_j]                    
                        annot_df.loc[elem_j, elem_i] = \
                            annot_df.loc[elem_i, elem_j]
                    else:
                        corr_matrix_df.loc[elem_j, elem_i] = 0
                        annot_df.loc[elem_j, elem_i] = ''
                        
                    
                else:
                    if data_df_clone is not None:
                        elem_list = list(data_df_clone.columns)
                        if (elem_i in elem_list) and (elem_j in elem_list):
                            temp_df = data_df_clone.pivot_table(
                                values = column_to_aggregate,
                                index = elem_i,
                                columns = elem_j,
                                aggfunc = 'count',
                                fill_value = 0)
                            corr_matrix_df.loc[elem_i, elem_j] = \
                                sci.stats.contingency.association(
                                temp_df, 
                                method = corr_method_quantitative, 
                                correction='False')
                            annot_df.loc[elem_i, elem_j] = \
                                f'{coef_name_dict[corr_method_quantitative]} = '\
                                    + str(round(corr_matrix_df.loc[elem_i, \
                                        elem_j], DecPlace))
                            if lower_half:
                                corr_matrix_df.loc[elem_j, elem_i] = \
                                    corr_matrix_df.loc[elem_i, elem_j]
                                annot_df.loc[elem_j, elem_i] = \
                                    annot_df.loc[elem_i, elem_j]
                            else:
                                corr_matrix_df.loc[elem_j, elem_i] = 0
                                annot_df.loc[elem_j, elem_i] = ''
                                
                        else:
                            annot_df.loc[elem_i, elem_j] = 'no col in df clone'
                            #annot_df.loc[elem_j, elem_i] = annot_df.loc[elem_i, elem_j]                   
                    else:
                        annot_df.loc[elem_i, elem_j] = 'no df clone'
                        #annot_df.loc[elem_j, elem_i] = annot_df.loc[elem_i, elem_j]                   
        
    # построение графика
    fig, axes = plt.subplots(figsize=(graph_size))
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    sns.heatmap(
        corr_matrix_df,
        vmin = 0, vmax = 1,
        cbar = True,
        #center = True,
        #annot = True,
        annot = annot_df,
        cmap = colormap,
        #fmt = '.4f',
        fmt = '',
        xticklabels = variable_list,
        yticklabels = variable_list,
        annot_kws = {"fontsize": annot_fontsize})
    axes.tick_params(labelsize = tick_fontsize)
    axes.figure.axes[-1].yaxis.set_tick_params(labelsize = colorbar_fontsize)
    plt.show()      
    
    # вывод графика
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    # вывод результата
    if corr_table_output:
        result = corr_matrix_df
        return result
    else:
        return     




#------------------------------------------------------------------------------
#   Функция graph_hist_boxplot_barplot_skew_kurt_XY
#   
#   Комплексная визуализация влияния качественного признака на количественный
#------------------------------------------------------------------------------

def graph_hist_boxplot_barplot_skew_kurt_XY(
    data_df:                  pd.core.frame.DataFrame,
    column_qualitative:       str,
    column_quantitative:      str,
    Y_min_histplot            = None, 
    Y_max_histplot            = None,
    Y_min_barplot             = None, 
    Y_max_barplot             = None,
    errorbar_mean             = 'sd',    # 'ci', 'pi', "se", 'sd' or None
    errorbar_median           = 'mad',
    title_figure:             str = None, 
    title_figure_fontsize:    int = 10,
    title_axes:               str = None, 
    title_axes_fontsize:      int = 14,
    title_graphics:           list = None,
    label_fontsize:           int = 10, 
    tick_fontsize:            int = 8, 
    label_legend_fontsize:    int = 10,
    palette                   = 'husl',
    table_output:             bool = True,
    p_level:                  float = 0.95,
    graph_size:               tuple = (210/INCH, 297/INCH),
    file_name:                str = None):
    
    """
    Комплексная визуализация влияния качественного признака на количественный, 
    включает следующие графики:
        - гистограмма;
        - коробчатая диаграмма;
        - столбчатые графики среднего значения и медианы с доверительными 
        интервалами (в виде полосы ошибок);
        - точечные графики показателей асимметрии и эксцесса с доверительными 
        интервалами (в виде полосы ошибок).

    Args:
        data_df (pd.core.frame.DataFrame):           
            массив исходных данных.
            
        column_qualitative (str):                    
            имя количественного признака в DataFrame.
            
        column_quantitative (str):                   
            имя качественного признака в DataFrame.
            
        Y_min_histplot, Y_max_histplot (optional):   
            мин. и макс. значение количественного признака на графике 
            гистограмм.      
            Defaults to None.
            
        Y_min_barplot, Y_max_barplot (optional):     
            мин. и макс. значение количественного признака на графике 
            коробчатой диаграммы.                                                 
            Defaults to None.
            
        errorbar_mean (str, optional):               
            параметр полосы ошибок для среднего значения 
            {'sd', 'ci', 'pi', "se", 'sd', or None}, где:    
                'sd' - standard deviation                                                         
                'se' - standard error                                                         
                'pi' - percentile interval                                                         
                'ci' - confidence interval
            Defaults to 'sd'.
            
        errorbar_median (str, optional):             
            параметр полосы ошибок для медианы 
            {'sd', 'ci', 'pi', "se", 'sd', 'mad or None}.
            'mad' - median absolute deviation.      
            Defaults to 'mad'.          
            
        title_figure (str, optional):                 
            заголовок рисунка (Figure). 
            Defaults to None.
            
        title_figure_fontsize (int, optional):       
            размер шрифта заголовка рисунка (Figure). 
            Defaults to 10.
            
        title_axes (str, optional):                  
            заголовок области рисования (Axes). 
            Defaults to None.
            
        title_axes_fontsize (int, optional):         
            размер шрифта заголовка области рисования (Axes). 
            Defaults to 14.
            
        title_graphics (list, optional):             
            заголовки графиков ax3, ax4, ax5, ax6. 
            Defaults to None.                                                         
            
        label_fontsize (int, optional):              
            размер шрифта подписей осей. 
            Defaults to 10.
            
        tick_fontsize (int, optional):               
            размер шрифта меток. 
            Defaults to 8.
            
        label_legend_fontsize (int, optional):       
            размер шрифта легенды. 
            Defaults to 10.
            
        palette (optional):                          
            цветовая палитра. 
            Defaults to 'husl'.         
            
        table_output (bool, optional):               
            логический параметр: True/False - выводить/не выводить таблицу 
            со стат.характеристиками. 
            Defaults to True.
            
        p_level (float, optional):                   
            доверительная вероятность. 
            Defaults to 0.95.
            
        graph_size (tuple, optional):                
            размер графика в дюймах. 
            Defaults to (297/INCH, 420/INCH), 
                где константа INCH = 25.4 мм/дюйм.
        
        file_name (str, optional):                
            имя файла для сохранения на диске. 
            Defaults to None.
    
    Returns:
        result (pd.core.frame.DataFrame):
            датасет со статистическими характеристиками
            
    """
    
    # создаем датасет со статистическими характеристиками по группированным данным
    data_df_describe = data_df.groupby(column_quantitative)\
        [column_qualitative].agg(['describe', 'median', 'skew'])
       
    # создаем отдельные датасеты по группированным данным и сохраняем их 
    # в словаре data_df_group_dict
    # элементы словаря data_df_group_dict - отдельные датасеты
    column_quantitative_values_list = list(data_df_describe.index)
    data_df_group_dict = dict()
    for elem in column_quantitative_values_list:
        mask_temp = data_df[column_quantitative] == elem
        data_df_temp = pd.DataFrame(data_df[mask_temp])
        data_df_group_dict[elem] = data_df_temp
        
    # добавляем в датасет data_df_describe отдельные стат.характеристики
    data_df_describe['kurtosis'] = \
        [sps.kurtosis(data_df_group_dict[key][column_qualitative]) \
         for key in data_df_group_dict.keys()]
    data_df_describe['mad'] = \
        [sm.robust.scale.mad(data_df_group_dict[key][column_qualitative], c=1) \
         for key in data_df_group_dict.keys()]
    data_df_describe['CV'] = \
        data_df_describe['describe']['std'] / data_df_describe['describe']\
            ['mean']
    #display(data_df_describe)
          
    # создание рисунка (Figure) и области рисования (Axes)
    fig = plt.figure(figsize = graph_size, constrained_layout = True)
    gs = fig.add_gridspec(4, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[3, 0])
    ax6 = fig.add_subplot(gs[3, 1])
    
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    
    # заголовки графиков ax3, ax4, ax5, ax6
    title_graphics = title_graphics if title_graphics \
        else ['mean' if not errorbar_mean else 'mean' + ', ' \
              + errorbar_mean, 
              'median' if not errorbar_median else 'median' + ', ' \
                  + errorbar_median, 
              'skew, ci', 'kurtosis, ci']
    
    # границы по осям
    Y_min_list = data_df_describe['describe']['min'].values
    Y_max_list = data_df_describe['describe']['max'].values
    
    Y_min_calc_list = data_df_describe['describe']['mean'].values \
        - 0.5*data_df_describe['describe']['std'].values
    Y_max_calc_list = data_df_describe['describe']['mean'].values \
        + 0.5*data_df_describe['describe']['std'].values
    
    Y_min_calc = min(min(Y_min_list), min(Y_min_calc_list))
    Y_max_calc = max(max(Y_max_list), max(Y_max_calc_list))
        
    if not(Y_min_histplot) and not(Y_max_histplot):
        (Y_min_histplot, Y_max_histplot) = \
            (min(Y_min_list)*0.99, max(Y_max_list)*1.01)
    if not(Y_min_barplot) and not(Y_max_barplot):        
        (Y_min_barplot, Y_max_barplot) = (Y_min_calc*0.99, Y_max_calc*1.01)
    
    # гистограмма
    ax1.set_title(title_axes, fontsize = title_axes_fontsize)
    sns.histplot(
        x = column_qualitative,
        #y = 'Name',
        data = data_df,
        # выбор вида гистограммы 
        # ('count', 'frequency', 'probability', 'percent', 'density')
        stat = 'count',      
        # выбор числа интервалов 
        # ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'))
        bins = 'auto',       
        # оценка плотности ядра (актуально только для одномерных данных)
        kde = True,          
        # визуальное представление (актуально только для одномерных данных): 
        # 'bars', 'step' - гистограмма, 'poly' - полигон
        element = 'step',    
        hue = column_quantitative,
        #multiple = layer,
        legend = True,
        label = '',
        palette = palette,
        #color = [sns.color_palette(palette).as_hex()[0:1]],
        ax = ax1)
    ax1.set_xlim(Y_min_histplot, Y_max_histplot)
    ax1.grid()
    ax1.tick_params(labelsize = tick_fontsize)
    plt.setp(ax1.get_legend().get_texts(), fontsize = label_legend_fontsize) 
        
    # коробчатая диаграмма
    #ax2.set_title('', fontsize = title_axes_fontsize)
    sns.boxplot(
        x = column_qualitative,
        y = column_quantitative,
        data = data_df,  
        #hue = column_quantitative,  
        orient = 'h',
        width = 0.5,
        palette = palette,
        #color = sns.palplot(sns.color_palette(palette)),
        ax = ax2)
    ax2.set_xlim(Y_min_histplot, Y_max_histplot)
    ax2.grid(axis = 'x')
    ax2.tick_params(labelsize = tick_fontsize)
    ax2.set_ylabel('')
    
    # среднее значение
    ax3.set_title(title_graphics[0], fontsize = title_axes_fontsize)
    sns.barplot(
        x = column_quantitative,
        y = column_qualitative,
        data = data_df,
        estimator = 'mean',
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html
        errorbar = errorbar_mean,    
        #ci = errorbar_mean,
        orient = 'v',
        width = 0.3,
        palette = palette,
        ax = ax3)
    ax3.set_ylabel(column_qualitative)
    ax3.set_ylim(Y_min_barplot, Y_max_barplot)
    #ax3.errorbar(errorbar_mean, 0)
    ax3.tick_params(labelsize = tick_fontsize)
    ax3.grid(axis = 'y')

    # медиана
    ax4.set_title(title_graphics[1], fontsize = title_axes_fontsize)
    sns.barplot(
        x = column_quantitative,
        y = column_qualitative,
        data = data_df,
        estimator = 'median',
        errorbar = errorbar_median if errorbar_median != 'mad' else None,
        #ci = errorbar_median,
        orient = 'v',
        width = 0.3,
        palette = palette,
        ax = ax4)
    ax4.set_ylabel(column_qualitative)
    ax4.set_ylim(Y_min_barplot, Y_max_barplot)
    ax4.tick_params(labelsize = tick_fontsize)
    ax4.grid(axis = 'y')
        
    # медиана (errorbar)
    # https://seaborn.pydata.org/tutorial/error_bars.html
    if errorbar_median == 'mad':
        x_coords = np.arange(0, len(data_df_describe), 1)
        y_coords = data_df_describe['median'][column_qualitative]
        yerr = data_df_describe['mad']
        ax4.errorbar(
            x_coords, y_coords, yerr,
            ecolor = 'black',
            elinewidth = 2, 
            fmt = ' ')    
    
    # асимметрия
    ax5.set_title(title_graphics[2], fontsize = title_axes_fontsize)
    data5 = data_df_describe['skew'][column_qualitative]
    sns.pointplot(
        x = data_df_describe.index,
        y = data5,
        scale = 1,
        join = True,
        ax = ax5)
    ax5.set_ylabel('')
    ax5.tick_params(labelsize = tick_fontsize)
    ax5.grid(axis = 'y')
    
    # асимметрия (errorbar)
    x_coords = np.arange(0, len(data_df_describe), 1)
    y_coords = data5
    N_list = data_df_describe['describe']['count'].values
    # табл.значение квантиля норм.распр.
    u_p = sps.norm.ppf(p_level, 0, 1)    
    # абс.ошибка асимметрии
    abs_err_As_func = lambda N: sqrt(6*N*(N-1) / ((N-2)*(N+1)*(N+3)))    
    yerr = [abs_err_As_func(n)*u_p for n in N_list]
    
    ax5.errorbar(
            x_coords, y_coords, yerr,
            #ecolor = sns.color_palette(palette).as_hex()[0],
            elinewidth = 2, 
            #fmt = ' '
            )    

    # эксцесс
    ax6.set_title(title_graphics[3], fontsize = title_axes_fontsize)
    data6 = data_df_describe['kurtosis']
    sns.pointplot(
        x = data_df_describe.index,
        y = data6,
        scale = 1,
        join = True,
        ax = ax6)
    ax6.set_ylabel('')
    ax6.tick_params(labelsize = tick_fontsize)
    ax6.grid(axis = 'y')
    
    # эксцесс (errorbar)
    y_coords = data6
    # абс.ошибка эксцесса
    abs_err_Es_func = lambda N: sqrt(24*N*(N-1)**2 / ((N-3)*(N-2)*(N+3)*(N+5)))    
    yerr = [abs_err_Es_func(n)*u_p for n in N_list]
    
    ax6.errorbar(
            x_coords, y_coords, yerr,
            #ecolor = sns.color_palette(palette).as_hex()[0],
            elinewidth = 2, 
            #fmt = ' '
            )    

    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    if table_output:
        result = data_df_describe.copy()
        return result
    else:
        return 



#------------------------------------------------------------------------------
#   Функция graph_regression_plot_sns
#------------------------------------------------------------------------------

def graph_regression_plot_sns(
    X, Y,
    regression_model,
    Xmin=None, Xmax=None,
    Ymin=None, Ymax=None,
    display_residuals=False,
    title_figure=None, title_figure_fontsize=None,
    title_axes=None, title_axes_fontsize=None,
    x_label=None,
    y_label=None,
    label_fontsize=None, tick_fontsize=12, 
    label_legend_regr_model='', label_legend_fontsize=12,
    s=50, linewidth_regr_model=2,
    graph_size=None,
    file_name=None):
    
    X = np.array(X)
    Y = np.array(Y)
    Ycalc = Y - regression_model(X)
    
    if not(Xmin) and not(Xmax):
        Xmin=min(X)*0.99
        Xmax=max(X)*1.01
    if not(Ymin) and not(Ymax):
        Ymin=min(Y)*0.99
        Ymax=max(Y)*1.01       
    
    # график с остатками
    # ------------------
    if display_residuals:
        if not(graph_size):
            graph_size = (297/INCH, 420/INCH/1.5)
        if not(title_figure_fontsize):
            title_figure_fontsize = 18
        if not(title_axes_fontsize):            
            title_axes_fontsize=16
        if not(label_fontsize):                        
            label_fontsize=13
        if not(label_legend_fontsize):
            label_legend_fontsize=12
        fig = plt.figure(figsize=graph_size)
        fig.suptitle(title_figure, fontsize = title_figure_fontsize)
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
                       
        # фактические данные
        ax1.set_title(title_axes, fontsize = title_axes_fontsize)
        sns.scatterplot(
            x=X, y=Y,
            label='data',
            s=s,
            color='red',
            ax=ax1)
        ax1.set_xlim(Xmin, Xmax)
        ax1.set_ylim(Ymin, Ymax)
        ax1.axvline(x = 0, color = 'k', linewidth = 1)
        ax1.axhline(y = 0, color = 'k', linewidth = 1)
        #ax1.set_xlabel(x_label, fontsize = label_fontsize)
        ax1.set_ylabel(y_label, fontsize = label_fontsize)
        ax1.tick_params(labelsize = tick_fontsize)
                
        # график регрессионной модели
        nx = 100
        hx = (Xmax - Xmin)/(nx - 1)
        x1 = np.linspace(Xmin, Xmax, nx)
        y1 = regression_model(x1)
        sns.lineplot(
            x=x1, y=y1,
            color='blue',
            linewidth=linewidth_regr_model,
            legend=True,
            label=label_legend_regr_model,
            ax=ax1)
        ax1.legend(prop={'size': label_legend_fontsize})
        
        # график остатков
        ax2.set_title('Residuals', fontsize = title_axes_fontsize)
        ax2.set_xlim(Xmin, Xmax)
        #ax2.set_ylim(Ymin, Ymax)
        sns.scatterplot(
            x=X, y=Ycalc,
            #label='фактические данные',
            s=s,
            color='orange',
            ax=ax2)
        
        ax2.axvline(x = 0, color = 'k', linewidth = 1)
        ax2.axhline(y = 0, color = 'k', linewidth = 1)
        ax2.set_xlabel(x_label, fontsize = label_fontsize)
        ax2.set_ylabel(r'$ΔY = Y - Y_{calc}$', fontsize = label_fontsize)
        ax2.tick_params(labelsize = tick_fontsize)
    
    # график без остатков
    # -------------------
    else:
        if not(graph_size):
            graph_size = (297/INCH, 210/INCH)
        if not(title_figure_fontsize):
            title_figure_fontsize = 18
        if not(title_axes_fontsize):            
            title_axes_fontsize=16
        if not(label_fontsize):                        
            label_fontsize=14
        if not(label_legend_fontsize):
            label_legend_fontsize=12
        fig, axes = plt.subplots(figsize=graph_size)
        fig.suptitle(title_figure, fontsize = title_figure_fontsize)
        axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
        # фактические данные
        sns.scatterplot(
            x=X, y=Y,
            label='фактические данные',
            s=s,
            color='red',
            ax=axes)
    
        # график регрессионной модели
        nx = 100
        hx = (Xmax - Xmin)/(nx - 1)
        x1 = np.linspace(Xmin, Xmax, nx)
        y1 = regression_model(x1)
        sns.lineplot(
            x=x1, y=y1,
            color='blue',
            linewidth=linewidth_regr_model,
            legend=True,
            label=label_legend_regr_model,
            ax=axes)
    
        axes.set_xlim(Xmin, Xmax)
        axes.set_ylim(Ymin, Ymax)    
        axes.axvline(x = 0, color = 'k', linewidth = 1)
        axes.axhline(y = 0, color = 'k', linewidth = 1)
        axes.set_xlabel(x_label, fontsize = label_fontsize)
        axes.set_ylabel(y_label, fontsize = label_fontsize)
        axes.tick_params(labelsize = tick_fontsize)
        axes.legend(prop={'size': label_legend_fontsize})
        
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return  

    

#==============================================================================
#               6. ФУНКЦИИ ДЛЯ ИССЛЕДОВАНИЯ ЗАКОНОВ РАСПРЕДЕЛЕНИЯ
#==============================================================================  


#------------------------------------------------------------------------------
#   Функция norm_distr_check
#   
#   Проверка нормальности распределения: возвращает DataFrame, содержащий 
#   результаты проверки нормальности распределения с использованием различных
#   тестов
#------------------------------------------------------------------------------


def norm_distr_check (
        data,
        p_level:    float = 0.95):

    """
    Проверка нормальности распределения: возвращает DataFrame, содержащий 
    результаты проверки нормальности распределения с использованием различных
    тестов.
    

    Args:
        data:                                     
            исходный массив данных.
            
        p_level (float, optional):           
            доверительная вероятность. 
            Defaults to 0.95.

    Returns:
        result (pd.core.frame.DataFrame):
            результат 
    
    Notes:
        1. Функция реализует следующие тесты:
            - тест Шапиро-Уилка (Shapiro-Wilk test) (при 8 <= N <= 1000)
            - тест Эппса-Палли (Epps_Pulley_test) (при N >= 8)
            - тест Д'Агостино (K2-test)
            - тест Андерсона-Дарлинга (Anderson-Darling test)
            - тест Колмогорова-Смирнова (Kolmogorov-Smirnov test) (при N >= 50)
            - тест Лиллиефорса (Lilliefors’ test)
            - тест Крамера-Мизеса-Смирнова (Cramér-von Mises test) (при N >= 40)
            - тест Пирсона (хи-квадрат) (chi-square test) (при N >= 100)
            - тест Харке-Бера (Jarque-Bera tes) (при N >= 2000)
            - тест асимметрии (при N >= 8)
            - тест эксцесса (при N >= 20)            
            
        2. Функция требует наличия файла table\Epps_Pulley_test_table.csv, 
            который содержит табличные значения критерия Эппса-Палли.
            
    """    
    
    a_level = 1 - p_level
    X = np.array(data)
    N = len(X)
       
    # тест Шапиро-Уилка (Shapiro-Wilk test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
    if N >= 8:
        result_ShW = sci.stats.shapiro(X)
        s_calc_ShW = result_ShW.statistic
        a_calc_ShW = result_ShW.pvalue
        conclusion_ShW = 'gaussian distribution' if a_calc_ShW >= a_level \
            else 'not gaussian distribution'
    else:
        result_ShW = '-'
        s_calc_ShW = '-'
        a_calc_ShW = '-'
        conclusion_ShW = 'count less than 8'

    # тест Эппса-Палли (Epps_Pulley_test)
    сdf_beta_I = lambda x, a, b: sci.stats.beta.cdf(x, a, b, loc=0, scale=1)
    g_beta_III = lambda z, δ: δ*z / (1+(δ-1)*z)
    cdf_beta_III = \
        lambda x, θ0, θ1, θ2, θ3, θ4: \
            сdf_beta_I(g_beta_III((x - θ4)/θ3, θ2), θ0, θ1)
    
    θ_1 = (1.8645, 2.5155, 5.8256, 0.9216, 0.0008)    # для 15 < n < 50
    θ_2 = (1.7669, 2.1668, 6.7594, 0.91, 0.0016)    # для n >= 50
    
    if N >= 8 and N <= 1000:
        X_mean = X.mean()
        m2 = np.var(X, ddof = 0)
        # расчетное значение статистики критерия
        A = sqrt(2) * np.sum([exp(-(X[i] - X_mean)**2 / (4*m2)) 
                              for i in range(N)])
        B = 2/N * np.sum(
            [np.sum([exp(-(X[j] - X[k])**2 / (2*m2)) for j in range(0, k)]) 
             for k in range(1, N)])
        s_calc_EP = 1 + N / sqrt(3) + B - A
        # табличное значение статистики критерия
        Tep_table_df = pd.read_csv(
            filepath_or_buffer='table/Epps_Pulley_test_table.csv',
            sep=';',
            index_col='n')
        p_level_dict = {
            0.9:   Tep_table_df.columns[0],
            0.95:  Tep_table_df.columns[1],
            0.975: Tep_table_df.columns[2],
            0.99:  Tep_table_df.columns[3]}
        f_lin = sci.interpolate.interp1d(Tep_table_df.index, \
                                         Tep_table_df[p_level_dict[p_level]])
        critical_value_EP = float(f_lin(N))
        # проверка гипотезы
        if 15 < N < 50:
            a_calc_EP = 1 - cdf_beta_III(s_calc_EP, θ_1[0], θ_1[1], \
                                         θ_1[2], θ_1[3], θ_1[4])
            conclusion_EP = 'gaussian distribution' if a_calc_EP > a_level \
                else 'not gaussian distribution'            
        elif N >= 50:
            a_calc_EP = 1 - cdf_beta_III(s_calc_EP, θ_2[0], θ_2[1], \
                                         θ_2[2], θ_2[3], θ_2[4])
            conclusion_EP = 'gaussian distribution' if a_calc_EP > a_level \
                else 'not gaussian distribution'            
        else:
            a_calc_EP = ''              
            conclusion_EP = 'gaussian distribution' \
                if s_calc_EP <= critical_value_EP \
                    else 'not gaussian distribution'            
                
    elif N > 1000:
        s_calc_EP = '-'
        critical_value_EP = '-'
        a_calc_EP = '-'
        conclusion_EP = 'count more than 1000'
    else:
        s_calc_EP = '-'
        critical_value_EP = '-'
        a_calc_EP = '-'
        conclusion_EP = 'count less than 8'
    
    
    # тест Д'Агостино (K2-test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
    if N >= 8:
        result_K2 = sci.stats.normaltest(X)
        s_calc_K2 = result_K2.statistic
        a_calc_K2 = result_K2.pvalue
        conclusion_K2 = 'gaussian distribution' if a_calc_K2 >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_K2 = '-'
        a_calc_K2 = '-'
        conclusion_K2 = 'count less than 8'
    
    # тест Андерсона-Дарлинга (Anderson-Darling test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
    result_AD = sci.stats.anderson(X)
    df_AD = pd.DataFrame({
        'a_level (%)': result_AD.significance_level,
        'statistic': [result_AD.statistic 
                      for i in range(len(result_AD.critical_values))],
        'critical_value': result_AD.critical_values
        })
    statistic_AD = float(df_AD[df_AD['a_level (%)'] == \
        round((1 - p_level)*100, 1)]['statistic'])
    critical_value_AD = \
        float(df_AD[df_AD['a_level (%)'] == round((1 - p_level)*100, 1)]\
              ['critical_value'])
    conclusion_AD = 'gaussian distribution' \
        if statistic_AD < critical_value_AD else 'not gaussian distribution'
    
    # тест Колмогорова-Смирнова (Kolmogorov-Smirnov test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html#scipy.stats.kstest
    if N >= 50:
        result_KS = sci.stats.kstest(X, 'norm')
        s_calc_KS = result_KS.statistic
        a_calc_KS = result_KS.pvalue
        conclusion_KS = 'gaussian distribution' if a_calc_KS >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_KS = '-'
        a_calc_KS = '-'
        conclusion_KS = 'count less than 50'
        
    # тест Лиллиефорса (Lilliefors’ test)
    # https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.lilliefors.html
    from statsmodels.stats.diagnostic import lilliefors
    s_calc_L, a_calc_L = sm.stats.diagnostic.lilliefors(X, 'norm')
    conclusion_L = 'gaussian distribution' if a_calc_L >= a_level \
        else 'not gaussian distribution'  
    
    # тест Крамера-Мизеса-Смирнова (Cramér-von Mises test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises.html#scipy-stats-cramervonmises
    if N >= 40:
        result_CvM = sci.stats.cramervonmises(X, 'norm')
        s_calc_CvM = result_CvM.statistic
        a_calc_CvM = result_CvM.pvalue
        conclusion_CvM = 'gaussian distribution' if a_calc_CvM >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_CvM = '-'
        a_calc_CvM = '-'
        conclusion_CvM = 'count less than 40'
    
    # тест Пирсона (хи-квадрат) (chi-square test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html#scipy-stats-chisquare
    if N >= 100:
        ddof = 2    # поправка к числу степеней свободы 
                    # (число параметров распределения, оцениваемое по выборке)
        result_Chi2 = sci.stats.chisquare(X, ddof=ddof)
        s_calc_Chi2 = result_Chi2.statistic
        a_calc_Chi2 = result_Chi2.pvalue
        conclusion_Chi2 = 'gaussian distribution' if a_calc_Chi2 >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_Chi2 = '-'
        a_calc_Chi2 = '-'
        conclusion_Chi2 = 'count less than 100'
        
    # тест Харке-Бера (асимметрии и эксцесса) (Jarque-Bera tes)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html#scipy-stats-jarque-bera
    if N > 2000:
        result_JB = sci.stats.jarque_bera(X)
        s_calc_JB = result_JB.statistic
        a_calc_JB = result_JB.pvalue
        conclusion_JB = 'gaussian distribution' if a_calc_JB >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_JB = '-'
        a_calc_JB = '-'
        conclusion_JB = 'count less than 2000'
    
    # тест асимметрии
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html#scipy-stats-skewtest
    if N >= 8:
        result_As = sci.stats.skewtest(X)
        s_calc_As = result_As.statistic
        a_calc_As = result_As.pvalue
        conclusion_As = 'gaussian distribution' if a_calc_As >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_As = '-'
        a_calc_As = '-'
        conclusion_As = 'count less than 8'
     
    # тест эксцесса
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosistest.html#scipy-stats-kurtosistest
    if N > 20:
        result_Es = sci.stats.kurtosistest(X)
        s_calc_Es = result_Es.statistic
        a_calc_Es = result_Es.pvalue
        conclusion_Es = 'gaussian distribution' if a_calc_Es >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_Es = '-'
        a_calc_Es = '-'
        conclusion_Es = 'count less than 20'
    
    # Создадим DataFrame для сводки результатов    
    result = pd.DataFrame({
    'test': (
        'Shapiro-Wilk test',
        'Epps-Pulley test',
        "D'Agostino's K-squared test",
        'Anderson-Darling test',
        'Kolmogorov–Smirnov test',
        'Lilliefors test',
        'Cramér–von Mises test',
        'Chi-squared test',
        'Jarque–Bera test',
        'skewtest',
        'kurtosistest'),
    'p_level': (p_level),
    'a_level': (a_level),
    'a_calc': (
        a_calc_ShW,
        a_calc_EP,
        a_calc_K2,
        '',
        a_calc_KS,
        a_calc_L,
        a_calc_CvM,
        a_calc_Chi2,
        a_calc_JB,
        a_calc_As,
        a_calc_Es),
    'a_calc >= a_level': (
        a_calc_ShW >= a_level if N >= 8 else '-',
        a_calc_EP >= a_level if N > 15 and N <= 1000 else '-',
        a_calc_K2 >= a_level if N >= 8 else '-',
        '',
        a_calc_KS >= a_level if N >= 50 else '-',
        a_calc_L >= a_level,
        a_calc_CvM >= a_level if N >= 40 else '-',
        a_calc_Chi2 >= a_level if N >= 100 else '-',
        a_calc_JB >= a_level if N >= 2000 else '-',
        a_calc_As >= a_level if N >= 8 else '-',
        a_calc_Es >= a_level if N > 20 else '-'),
    'statistic': (
        s_calc_ShW,
        s_calc_EP,
        s_calc_K2,
        statistic_AD,
        s_calc_KS,
        s_calc_L,
        s_calc_CvM,
        s_calc_Chi2,
        s_calc_JB,
        s_calc_As,
        s_calc_Es),
    'critical_value': (
        '',
        critical_value_EP,
        '',
        critical_value_AD,
        '', '', '', '', '', '', ''),
    'statistic < critical_value': (
        '',
        s_calc_EP < critical_value_EP  if N >= 8 else '-',
        '',
        statistic_AD < critical_value_AD,
        '', '', '', '', '', '', ''),
    'conclusion': (
        conclusion_ShW,
        conclusion_EP,
        conclusion_K2,
        conclusion_AD,
        conclusion_KS,
        conclusion_L,
        conclusion_CvM,
        conclusion_Chi2,
        conclusion_JB,
        conclusion_As,
        conclusion_Es
        )})  
        
    return result



#==============================================================================
#               7. ФУНКЦИИ ДЛЯ ВЫЯВЛЕНИЯ АНОМАЛЬНЫХ ЗНАЧЕНИЙ (ВЫБРОСОВ)
#==============================================================================    
    
#------------------------------------------------------------------------------
#   Функция detecting_outliers_mad_test
#   Критерий максимального абсолютного отклонения (М.А.О.) (см.[Кобзарь, с.547]
#------------------------------------------------------------------------------

def detecting_outliers_mad_test(
        X_in,
        p_level=0.95,
        show_in_full=False,
        delete_outlier=False):
    N = len(X_in)
    X_mean = X_in.mean()
    X_std = X_in.std(ddof = 1)
    # создаем вспомогательный DataFrame, отсортированный по возрастанию
    data = pd.DataFrame({'value': X_in}).sort_values(by='value')
    # расчетное значение статистики критерия М.А.О.
    data['mad_calc'] = abs((data['value'] - X_mean)/X_std)
    # аппроксимация табличного значения статистики критерия М.А.О.
    mad_table = lambda n: 1.39 + 0.462*log(n-3) if 5<=n<35 else 2.136 - 0.281*log(n-15) if 35<=n<=500 else '-'
    data['mad_table'] = mad_table(N)
    # добавляем во вспомогательный DataFrame столбец 'outlier_conclusion', в котором выбросу будут отмечены как 'outlier'
    data['outlier_conclusion'] = '-'
    mask = data['mad_calc'] >= data['mad_table']
    data.loc[mask, 'outlier_conclusion'] = 'outlier'
        
    if delete_outlier:
        X_out = np.delete(X_in, [data.loc[mask].index])
        return data, X_out
    else:
        return data






#==============================================================================
#               8. ФУНКЦИИ ДЛЯ ИССЛЕДОВАНИЯ КАТЕГОРИАЛЬНЫХ ДАННЫХ
#==============================================================================  

#------------------------------------------------------------------------------
#   Функция graph_contingency_tables_hist_3D
#   
#   Визуализация категориальных данных: построение трехмерных гистограмм
#------------------------------------------------------------------------------

def graph_contingency_tables_hist_3D(
    data_df_in:               pd.core.frame.DataFrame = None,
    data_XY_list_in:          list = None,
    title_figure:             str = None, 
    title_figure_fontsize:    int = 10,
    title_axes:               str = None, 
    title_axes_fontsize:      int = 12,
    rows_label                = None, 
    cols_label                = None, 
    vertical_label:           str = None, 
    label_fontsize:           int = 10, 
    rows_ticklabels_list:     list = None, 
    cols_ticklabels_list:     list = None,
    tick_fontsize:            int = 8, 
    rows_tick_rotation:       int = 47.5, 
    cols_tick_rotation:       int = -17.5, 
    legend:                   str = None, 
    legend_fontsize:          int = 10,
    labelpad:                 int = 20,
    color                     = None,
    tight_layout:             bool = True,
    graph_size:               tuple = (297/INCH, 297/INCH/2),
    file_name:                str = None):
    
    """
    Визуализация категориальных данных: построение трехмерных гистограмм.

    Args:
        data_df_in (pd.core.frame.DataFrame, optional): 
            массив исходных данных. 
            Defaults to None.
            
        data_XY_list_in (list, optional):               
            массив исходных данных. 
            Defaults to None.
            
        title_figure (str, optional):             
            заголовок рисунка (Figure). 
            Defaults to None.
            
        title_figure_fontsize (int, optional):    
            размер шрифта заголовка рисунка (Figure). 
            Defaults to 10.
            
        title_axes (str, optional):               
            заголовок области рисования (Axes). 
            Defaults to None.
        
        title_axes_fontsize (int, optional):      
            размер шрифта заголовка области рисования (Axes). 
            Defaults to 14.
            
        rows_label (optional):                  
            подпись оси (по строкам). 
            Defaults to None.
            
        cols_label (optional):                  
            подпись оси (по столбцам). 
            Defaults to None.
            
        vertical_label (str, optional):              
            подпись вертикальной оси. 
            Defaults to None.
            
        label_fontsize (int, optional):                 
            размер шрифта подписей осей. 
            Defaults to 10.
            
        rows_ticklabels_list (list, optional):        
            список меток для оси (по строкам). 
            Defaults to None.
            
        cols_ticklabels_list (list, optional):        
            список меток для оси (по столбцам). 
            Defaults to None.
            
        tick_fontsize (int, optional):                  
            размер шрифта меток осей. 
            Defaults to 8.
            
        rows_tick_rotation (int, optional):             
            угол поворота меток для оси (по строкам). 
            Defaults to 47.5.
            
        cols_tick_rotation (int, optional):             
            угол поворота меток для оси (по столбцам). 
            Defaults to -17.5.
            
        legend (str, optional):                      
            текст легенды. 
            Defaults to None.
            
        legend_fontsize (int, optional):                
            размер шрифта легенды. 
            Defaults to 10.
            
        labelpad (int, optional):                       
            расстояние между осью и метками. 
            Defaults to 20.
            
        color (optional):                       
            цвет графика. 
            Defaults to None.
            
        tight_layout (bool, optional):                  
            логический параметр: автоматическая настройка плотной компоновки 
            графика (да/нет, True/False). 
            Defaults to True.
        
        graph_size (tuple, optional):             
            размер графика в дюймах. 
            Defaults to (210/INCH, 297/INCH/2), 
                где константа INCH = 25.4 мм/дюйм.
        
        file_name (str, optional):                
            имя файла для сохранения на диске. 
            Defaults to None.
           
    Returns:
        None
            
    """
    
    
    # создание рисунка (Figure) и области рисования (Axes)
    fig = plt.figure(figsize=graph_size)
    ax = plt.axes(projection = "3d")
    #ax = fig.gca(projection='3d')
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    ax.set_title(title_axes, fontsize = title_axes_fontsize)
        
    # данные для построения графика
    if data_df_in is not None:
        data = np.array(data_df_in)
        NumOfCols = data_df_in.shape[1]
        NumOfRows = data_df_in.shape[0]
    else:
        data = np.array(data_XY_list_in)
        NumOfCols = np.shape(data)[1]
        NumOfRows = np.shape(data)[0]
                
    # координаты точки привязки столбцов
    xpos = np.arange(0, NumOfCols, 1)
    ypos = np.arange(0, NumOfRows, 1)
        
    # формируем сетку координат
    xpos, ypos = np.meshgrid(xpos + 0.5, ypos + 0.5)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    
    # инициализируем для zpos нулевые значение, чтобы столбцы начинались с нуля
    zpos = np.zeros(NumOfCols * NumOfRows)
        
    # формируем ширину и глубину столбцов
    dx = np.ones(NumOfRows * NumOfCols) * 0.5
    dy = np.ones(NumOfCols * NumOfRows) * 0.5
        
    # формируем высоту столбцов
    dz = data.flatten()
            
    # построение трехмерного графика
    if not color:
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
    else:
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color)
            
    # подписи осей
    x_label = cols_label if cols_label else ''
    y_label = rows_label if rows_label else ''
    z_label = vertical_label if vertical_label else ''
    
    ax.set_xlabel(x_label, fontsize = label_fontsize)
    ax.set_ylabel(y_label, fontsize = label_fontsize)
    ax.set_zlabel(z_label, fontsize = label_fontsize)
    
    # метки осей
    x_ticklabels_list = cols_ticklabels_list if cols_ticklabels_list \
        else list(data_df_in.columns) if (data_df_in is not None) else ''
    y_ticklabels_list = rows_ticklabels_list if rows_ticklabels_list \
        else list(data_df_in.index) if data_df_in is not None else ''
    
    # форматирование меток осей (https://matplotlib.org/stable/api/ticker_api.html)
    from matplotlib.ticker import IndexLocator
    ax.xaxis.set_major_locator(IndexLocator(1.0, 0.25))
    ax.yaxis.set_major_locator(IndexLocator(1.0, 0.25))
            
    # устанавливаем метки осей
    ax.set_xticklabels(x_ticklabels_list, fontsize = tick_fontsize, \
        rotation=cols_tick_rotation)
    ax.set_yticklabels(y_ticklabels_list, fontsize = tick_fontsize, \
        rotation=rows_tick_rotation)
        
    # расстояние между подписями осей и метками осей
    ax.xaxis.labelpad = labelpad
    ax.yaxis.labelpad = labelpad
    
    # легенда
    if legend:
        b1 = plt.Rectangle((0, 0), 1, 1)
        ax.legend([b1], [legend], prop={'size': legend_fontsize})
        
    # автоматическая настройка плотной компоновки графика
    if tight_layout:
        fig.tight_layout()
        
    # вывод графика
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return



#------------------------------------------------------------------------------
#   Функция graph_contingency_tables_mosaicplot_sm
#   
#   Визуализация категориальных данных: построение мозаичных диаграмм
#------------------------------------------------------------------------------

def graph_contingency_tables_mosaicplot_sm(
    data_df_in:              pd.core.frame.DataFrame = None,
    data_XY_list_in:         list = None,
    properties:              dict = {},
    labelizer:               bool = True,
    title_figure             = None, 
    title_figure_fontsize    = 10,
    title_axes               = None, 
    title_axes_fontsize      = 14,
    x_label:                 str = None, 
    y_label:                 str = None, 
    label_fontsize           = 10, 
    x_ticklabels_list:       list = None, 
    y_ticklabels_list:       list = None,
    x_ticklabels:            bool = True, 
    y_ticklabels:            bool = True,
    tick_fontsize:           int = 8,
    tick_label_rotation:     int = 0,
    legend_list:             list = None, 
    legend_fontsize:         int = 10,
    text_fontsize:           int = 12,
    gap:                     float = 0.005,
    horizontal:              bool = True,
    statistic:               bool = True,
    tight_layout:            bool = True,
    graph_size:              tuple = (210/INCH, 297/INCH/2),
    file_name:               str = None):
    
    """
    Визуализация категориальных данных: построение мозаичных диаграмм.

    Args:
        data_df_in (pd.core.frame.DataFrame, optional): 
            массив исходных данных. 
            Defaults to None.
            
        data_XY_list_in (list, optional):               
            массив исходных данных. 
            Defaults to None.
            
        properties (dict, optional):                    
            словарь, задающий свойства плиток графика (цвет, штриховка и пр.). 
            Defaults to {}.
            
        labelizer (bool, optional):
            логический параметр: True/False - отображать/не отображать
            текст в центре каждой плитки графика. 
            Defaults to True.
            
        title_figure (str, optional):             
            заголовок рисунка (Figure). 
            Defaults to None.
            
        title_figure_fontsize (int, optional):    
            размер шрифта заголовка рисунка (Figure). 
            Defaults to 10.
            
        title_axes (str, optional):               
            заголовок области рисования (Axes). 
            Defaults to None.
        
        title_axes_fontsize (int, optional):      
            размер шрифта заголовка области рисования (Axes). 
            Defaults to 14.
            
        x_label (str, optional):
            подпись оси OX. 
            Defaults to None.
            
        y_label (str, optional):                     
            подпись оси OY. 
            Defaults to None.
            
        label_fontsize (int, optional):                 
            размер шрифта подписей. 
            Defaults to 10.
            
        x_ticklabels_list (list, optional):           
            список меток для оси OX. 
            Defaults to None.
            
        y_ticklabels_list (list, optional):           
            список меток для оси OY. 
            Defaults to None.
            
        x_ticklabels (bool, optional):
            логический параметр: True/False - отображать на графике (да/нет) 
            метки для оси OX. 
            Defaults to True.
            
        y_ticklabels (bool, optional):
            логический параметр: True/False - отображать на графике (да/нет) 
            метки для оси OY. 
            Defaults to True.
            
        tick_fontsize (int, optional):
            размер шрифта меток осей. 
            Defaults to 8.
            
        tick_label_rotation (int, optional):            
            угол поворота меток для оси. 
            Defaults to 0.
            
        legend_list (list, optional):                 
            список названий категорий для легенды. 
            Defaults to None.
            
        legend_fontsize (int, optional):                
            размер шрифта легенды. 
            Defaults to 10.
            
        text_fontsize (int, optional):                  
            размер шрифта подписей в центре плиток графика. 
            Defaults to 12.
            
        gap (float, optional):                          
            список зазоров. Defaults to 0.005.
            
        horizontal (bool, optional):                    
            логический параметр: начальное направление разделения. 
            Defaults to True.
            
        statistic (bool, optional):                     
            логический параметр: True/False - применять статистическую модель 
            для придания цвета графику (да/нет). 
            Defaults to True.
            
        tight_layout (bool, optional):                  
            логический параметр: автоматическая настройка плотной компоновки 
            графика (да/нет, True/False). 
            Defaults to True.
        
        graph_size (tuple, optional):             
            размер графика в дюймах. 
            Defaults to (210/INCH, 297/INCH/2), 
                где константа INCH = 25.4 мм/дюйм.
        
        file_name (str, optional):                
            имя файла для сохранения на диске. 
            Defaults to None.

    Returns:
        None
            
    """
    
    # создание рисунка (Figure) и области рисования (Axes)
    fig, axes = plt.subplots(figsize=graph_size)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    # данные для построения графика
    if data_df_in is not None:
        data_df = data_df_in.copy()
        if x_ticklabels_list:
            data_df = data_df.set_index(pd.Index(x_ticklabels_list))
    else:
        data_df = pd.DataFrame(data_XY_list_in)
        if x_ticklabels_list:
            data_df = data_df.set_index(pd.Index(x_ticklabels_list))
        if y_ticklabels_list:
            data_df.columns = y_ticklabels_list
        
    data_np = np.array(data_XY_list_in) if data_XY_list_in is not None \
        else np.array(data_df_in)
    
    # установка шрифта подписей в теле графика 
    if text_fontsize:
        plt.rcParams["font.size"] = text_fontsize
            
    # метки осей
    if data_df is not None:
        x_list = list(map(str, x_ticklabels_list)) if x_ticklabels_list \
            else list(map(str, data_df.index))
        y_list = list(map(str, y_ticklabels_list)) if y_ticklabels_list \
            else list(map(str, data_df.columns))
    else:
        x_list = list(map(str, x_ticklabels_list)) if x_ticklabels_list \
            else list(map(str, range(data_np.shape[0])))
        y_list = list(map(str, y_ticklabels_list)) if y_ticklabels_list \
            else list(map(str, range(data_np.shape[1])))
        
    if not labelizer:
        if not x_ticklabels:
            axes.tick_params(axis='x', colors='white')
        if not y_ticklabels:
            axes.tick_params(axis='y', colors='white')

    # подписи осей
    x_label = x_label if x_label else ''
    y_label = y_label if y_label else ''
        
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)            
            
                
    # формируем словарь (dict) data
    data_dict = {}
    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            data_dict[(x, y)] = data_np[i, j]
                
    # формируем словарь (dict) labelizer и функцию labelizer_func
    labelizer_dict = {}
    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            labelizer_dict[(x, y)] = data_np[i, j] if labelizer else ''
    labelizer_func = lambda k: labelizer_dict[k]  
    
    # построение графика
    from statsmodels.graphics.mosaicplot import mosaic
    mosaic(data_dict,
           title=title_axes,
           statistic=statistic,
           ax=axes,
           horizontal=horizontal,
           gap=gap,
           label_rotation=tick_label_rotation,
           #axes_label=False,
           labelizer=labelizer_func,
           properties=properties)
    axes.tick_params(labelsize = tick_fontsize)
            
    # легенда
    if legend_list:
        axes.legend(legend_list,
                    bbox_to_anchor=(1, 0.5),
                    loc="center left",
                    #mode="expand",
                    ncol=1)
        
    # автоматическая настройка плотной компоновки графика
    if tight_layout:
        fig.tight_layout()
        
    # вывод графика
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
    
    # возврат размера шрифта подписей в теле графика по умолчанию
    if text_fontsize:
        plt.rcParams["font.size"] = 10
            
    return



#------------------------------------------------------------------------------
#   Функция make_color_mosaicplot_dict
#   
#   Визуализация категориальных данных: вспомогательная функция - формирует 
#   словарь (dict) свойств плиток мозаичного графика (цвет, штриховка и пр.) 
#   для функции graph_contingency_tables_mosaicplot_sm.
#------------------------------------------------------------------------------

def make_color_mosaicplot_dict(
        rows_list:          list, 
        cols_list:          list, 
        props_dict_rows:    dict = None,
        props_dict_cols:    dict = None):
    
    """
    Визуализация категориальных данных: вспомогательная функция - формирует 
    словарь свойств плиток мозаичного графика (цвет, штриховка и пр.) 
    для функции graph_contingency_tables_mosaicplot_sm.

    Args:
        rows_list (list):                 
            список категорий (по строкам)
            
        cols_list (list):                 
            список категорий (по столбцам)
            
        props_dict_rows (dict, optional): 
            словарь цветовых свойств категорий (по строкам). 
            Defaults to None.
            
        props_dict_cols (dict, optional): 
            словарь цветовых свойств категорий (по столбцам). 
            Defaults to None.

    Returns:
        result (dict): 
            словарь свойств плиток мозаичного графика (цвет, штриховка и пр.) 
            для функции graph_contingency_tables_mosaicplot_sm.
            
    """
    
    result = {}
    rows = list(map(str, rows_list))
    cols = list(map(str, cols_list))
    
    if props_dict_rows:
        for col in cols:
            for row in rows:
                result[(col, row)] = {'facecolor': props_dict_rows[row]}
    
    if props_dict_cols:
        for col in cols:
            for row in rows:
                result[(col, row)] = {'facecolor': props_dict_cols[col]}
    
    return result    



#------------------------------------------------------------------------------
#   Функция graph_contingency_tables_bar_freqint
#   
#   Визуализация категориальных данных: построение столбчатых диаграмм и 
#   графиков взаимодействия частот
#------------------------------------------------------------------------------

def graph_contingency_tables_bar_freqint(
    data_df_in:               pd.core.frame.DataFrame = None,
    data_XY_list_in:          list = None,
    graph_inclusion:          str = 'arf',
    title_figure:             str = None, 
    title_figure_fontsize:    int = 12, 
    title_axes_fontsize:      int = 10,
    x_label                   = None, 
    y_label                   = None, 
    label_fontsize:           int = 10, 
    x_ticklabels_list:        list = None, 
    y_ticklabels_list:        list = None, 
    tick_fontsize:            int = 8,
    legend_fontsize:          int = 10,
    color                     = None,
    tight_layout:             bool = True,
    result_output:            bool = False,
    graph_size:               tuple = None,
    file_name:                str = None):
    
    """
    Визуализация категориальных данных: построение столбчатых диаграмм и 
    графиков взаимодействия частот.

    Args:
        data_df_in (pd.core.frame.DataFrame, optional): 
            массив исходных данных. 
            Defaults to None.
            
        data_XY_list_in (list, optional):               
            массив исходных данных. 
            Defaults to None.
            
        graph_inclusion (str, optional):                
            параметр, определяющий набор графиков на одном рисунке (Figure).                                                  
            Defaults to 'arf', где:                                                            
                'a' - столбчатая диаграмма (в абсолютных частотах)
                'r' - столбчатая диаграмма (в относительных частотах)
                'f' - график взаимодействия частот
                                                            
        title_figure (str, optional):             
            заголовок рисунка (Figure). 
            Defaults to None.
            
        title_figure_fontsize (int, optional):    
            размер шрифта заголовка рисунка (Figure). 
            Defaults to 14.
            
        title_axes_fontsize (int, optional):      
            размер шрифта заголовка области рисования (Axes). 
            Defaults to 12.
            
        x_label (str, optional):
            подпись оси OX. 
            Defaults to None.
            
        y_label (str, optional):                     
            подпись оси OY. 
            Defaults to None.
            
        label_fontsize (int, optional):                 
            размер шрифта подписей. 
            Defaults to 10.
            
        x_ticklabels_list (list, optional):           
            список меток для оси OX. 
            Defaults to None.
            
        y_ticklabels_list (list, optional):           
            список меток для оси OY. 
            Defaults to None.
            
        tick_fontsize (int, optional):
            временно заблокировано. Defaults to 8.
        
        legend_fontsize (int, optional):                
            размер шрифта легенды. 
            Defaults to 10.
                    
            
        color (optional):                       
            список, задающий цвета для категорий. 
            Defaults to None.
            
        tight_layout (bool, optional):                  
            логический параметр: автоматическая настройка плотной компоновки 
            графика (да/нет, True/False). 
            Defaults to True.
            
        result_output (bool, optional):                 
            логический параметр: True/False - выводить таблицу (DataFrame) 
            c числовыми данными (да/нет).
            Defaults to False.
        
        graph_size (tuple, optional):             
            размер графика в дюймах. 
            Defaults to None.
        
        file_name (str, optional):                
            имя файла для сохранения на диске. 
            Defaults to None.

    Returns:
        data_df_abs, data_df_rel (pd.core.frame.DataFrame, optional):
            таблица (DataFrame) с абс. и отн. частотами
            
    """
    
    # данные для построения графика
    if data_df_in is not None:
        data_df_abs = data_df_in.copy()
        if x_ticklabels_list:
            data_df_abs = data_df_abs.set_index(pd.Index(x_ticklabels_list))
        if y_ticklabels_list:
            data_df_abs.columns = y_ticklabels_list
    else:
        data_df_abs = pd.DataFrame(data_XY_list_in)
        if x_ticklabels_list:
            data_df_abs = data_df_abs.set_index(pd.Index(x_ticklabels_list))
        if y_ticklabels_list:
            data_df_abs.columns = y_ticklabels_list
    data_df_rel = None
    
    data_np = np.array(data_XY_list_in) if data_XY_list_in is not None \
        else np.array(data_df_in)
    
    # определение формы и размеров области рисования (Axes)
    count_graph = len(graph_inclusion)    # число графиков
    ax_rows = 1
    ax_cols = count_graph    # размерность области рисования (Axes)
    
    # создание рисунка (Figure) и области рисования (Axes)
    graph_size_dict = {
        1: (210/INCH*0.6, 297/INCH/2),
        2: (210/INCH*1.2, 297/INCH/2),
        3: (210/INCH*1.8, 297/INCH/2)}
    
    if not(graph_size):
        graph_size = graph_size_dict[count_graph]
    
    fig = plt.figure(figsize=graph_size)
    
    if count_graph == 3:
        ax1 = plt.subplot(1,3,1)
        ax2 = plt.subplot(1,3,2)
        ax3 = plt.subplot(1,3,3)
    elif count_graph == 2:
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
    elif count_graph == 1:
        ax1 = plt.subplot(1,1,1)
        
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    
    # столбчатая диаграмма (абсолютные частоты)
    if 'a' in graph_inclusion:
        if color:
            data_df_abs.plot.bar(
                color = color,
                stacked=True,
                rot=0,
                legend=True,
                ax=ax1)
        else:
            data_df_abs.plot.bar(
                #color = color_list,
                stacked=True,
                rot=0,
                legend=True,
                ax=ax1)
        ax1.legend(loc='best', fontsize = legend_fontsize, \
                   title=data_df_abs.columns.name)
        ax1.set_title('Absolute values', fontsize=title_axes_fontsize)
        ax1.set_xlabel(x_label, fontsize = label_fontsize)
        ax1.set_ylabel(y_label, fontsize = label_fontsize)
        ax1.tick_params(labelsize = tick_fontsize)
        ax1.yaxis.grid()
            
    # столбчатая диаграмма (относительные частоты)
    if 'r' in graph_inclusion:
        data_df_rel = data_df_abs.copy()
        sum_data = np.sum(data_np)
        data_df_abs.sum(axis=1)
        
        for col in data_df_rel.columns:
            data_df_rel[col] = data_df_rel[col] / data_df_abs.sum(axis=1)
        
        if color:
            data_df_rel.plot.bar(
                color = color,
                stacked=True,
                rot=0,
                legend=True,
                ax = ax1 if ((graph_inclusion == 'r') or 
                             (graph_inclusion == 'rf')) else ax2,
                alpha = 0.5)
        else:
            data_df_rel.plot.bar(
                #color = color,
                stacked=True,
                rot=0,
                legend=True,
                ax = ax1 if ((graph_inclusion == 'r') or 
                             (graph_inclusion == 'rf')) else ax2,
                alpha = 0.5)
        
        if (graph_inclusion == 'r') or (graph_inclusion == 'rf'):
            ax1.legend(loc='best', fontsize = legend_fontsize, \
                       title=data_df_abs.columns.name)
            ax1.set_title('Relative values', fontsize=title_axes_fontsize)
            ax1.set_xlabel(x_label, fontsize = label_fontsize)
            ax1.set_ylabel(y_label, fontsize = label_fontsize)
            ax1.tick_params(labelsize = tick_fontsize)
            ax1.yaxis.grid()
        else:
            ax2.legend(loc='best', fontsize = legend_fontsize, \
                       title=data_df_abs.columns.name)
            ax2.set_title('Relative values', fontsize=title_axes_fontsize)
            ax2.set_xlabel(x_label, fontsize = label_fontsize)
            ax2.set_ylabel(y_label, fontsize = label_fontsize)
            ax2.tick_params(labelsize = tick_fontsize)
            ax2.yaxis.grid()
                            
    # график взаимодействия частот
    if 'f' in graph_inclusion:
        if color:
            sns.lineplot(
                data=data_df_abs,
                palette = color,
                dashes=False,
                lw=3,
                #markers=['o','o'],
                markersize=10,
                ax = ax1 if (graph_inclusion == 'f') else \
                    ax3 if (graph_inclusion == 'arf') else ax2)
        else:
            sns.lineplot(
                data=data_df_abs,
                #palette = color,
                dashes=False,
                lw=3,
                #markers=['o','o'],
                markersize=10,
                ax = ax1 if (graph_inclusion == 'f') else \
                    ax3 if (graph_inclusion == 'arf') else ax2)
        
        if (graph_inclusion == 'f'):
            ax1.legend(loc='best', fontsize = legend_fontsize, \
                       title=data_df_abs.columns.name)
            ax1.set_title('Graph of frequency interactions', \
                          fontsize=title_axes_fontsize)
            ax1.set_xlabel(x_label, fontsize = label_fontsize)
            ax1.set_ylabel(y_label, fontsize = label_fontsize)
            ax1.tick_params(labelsize = tick_fontsize)
            ax1.yaxis.grid()
        elif (graph_inclusion == 'arf'):
            ax3.legend(loc='best', fontsize = legend_fontsize, \
                       title=data_df_abs.columns.name)
            ax3.set_title('Graph of frequency interactions', \
                          fontsize=title_axes_fontsize)
            ax3.set_xlabel(x_label, fontsize = label_fontsize)
            ax3.set_ylabel(y_label, fontsize = label_fontsize)
            ax3.tick_params(labelsize = tick_fontsize)
            ax3.yaxis.grid()
        else:
            ax2.legend(loc='best', fontsize = legend_fontsize, \
                       title=data_df_abs.columns.name)
            ax2.set_title('Graph of frequency interactions', \
                          fontsize=title_axes_fontsize)
            ax2.set_xlabel(x_label, fontsize = label_fontsize)
            ax2.set_ylabel(y_label, fontsize = label_fontsize)
            ax2.tick_params(labelsize = tick_fontsize)
            ax2.yaxis.grid()
    
    # автоматическая настройка плотной компоновки графика
    if tight_layout:
        fig.tight_layout()
    
    # вывод графика
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    # формирование и вывод результата
    if result_output:
        data_df_abs['sum'] = data_df_abs.sum(axis=1)
        if data_df_rel is not None:
            data_df_rel['sum'] = data_df_rel.sum(axis=1)
            print('\nAbsolute values:')
            display(data_df_abs)
            print('\nRelative values:')
            display(data_df_rel)
        else:
            print('\nAbsolute values:')
            display(data_df_abs)
    
    return



#------------------------------------------------------------------------------
#   Функция graph_contingency_tables_heatmap
#   
#   Визуализация категориальных данных: построение графика тепловой карты
#------------------------------------------------------------------------------

def graph_contingency_tables_heatmap(
    data_df_in:               pd.core.frame.DataFrame = None,
    data_XY_list_in:          list = None,
    title_figure:             str = None, 
    title_figure_fontsize:    int = 10,
    title_axes:               str = None, 
    title_axes_fontsize:      int = 12,
    x_label                   = None, 
    y_label                   = None, 
    label_fontsize:           int = 10, 
    x_ticklabels_list:        list = None, 
    y_ticklabels_list:        list = None, 
    tick_fontsize:            int = 8,
    annot_fontsize:           int = 8,
    colorbar_fontsize:        int = 8,
    values_type:              str = 'absolute',
    color_map:                str = 'binary',
    robust:                   bool = False,
    fmt:                      str = '.0f',
    tight_layout:             bool =True,
    #result_output = False,
    graph_size:               tuple = (210/INCH, 297/INCH/2),
    file_name:                str = None):
    
    """
    Визуализация категориальных данных построение графика тепловой карты 
    (heatmap).

    Args:
        data_df_in (pd.core.frame.DataFrame, optional): 
            массив исходных данных. 
            Defaults to None.
            
        data_XY_list_in (list, optional):               
            массив исходных данных. 
            Defaults to None.
            
        title_figure (str, optional):             
            заголовок рисунка (Figure). 
            Defaults to None.
            
        title_figure_fontsize (int, optional):    
            размер шрифта заголовка рисунка (Figure). 
            Defaults to 10.
            
        title_axes (str, optional):               
            заголовок области рисования (Axes). 
            Defaults to None.
        
        title_axes_fontsize (int, optional):      
            размер шрифта заголовка области рисования (Axes). 
            Defaults to 12.
            
        x_label (str, optional):
            подпись оси OX. 
            Defaults to None.
            
        y_label (str, optional):                     
            подпись оси OY. 
            Defaults to None.
            
        label_fontsize (int, optional):                 
            временно заблокировано. Defaults to 10.
            
        x_ticklabels_list (list, optional):           
            список меток для оси OX. 
            Defaults to None.
            
        y_ticklabels_list (list, optional):           
            список меток для оси OY. 
            Defaults to None.
            
        tick_fontsize (int, optional):
            размер шрифта меток осей. 
            Defaults to 8.
            
        annot_fontsize (int, optional):       
            размер шрифта подписей в центре плиток графика. 
            Defaults to 8.
            
        colorbar_fontsize (int, optional):       
            размер шрифта подписей цветовой полосы. 
            Defaults to 8.
            
        values_type (str, optional):                    
            параметр, задающий в каких частотах строится график:     
                абсолютные/относительные, absolute/relative.     
                Defaults to 'absolute'.
                
        color_map (str, optional):                      
            цветовая карта (colormap) для графика. 
            Defaults to 'binary'.
            
        robust (bool, optional):                        
            логический параметр: если True и vmin или vmax отсутствуют, 
            диапазон цветовой карты вычисляется с надежными квантилями 
            вместо экстремальных значений. 
            Defaults to False.
            
        fmt (str, optional):                            
            числовой формат подписей в центре плиток графика. 
            Defaults to '.0f'.
        
        tight_layout (bool, optional):                  
            логический параметр: автоматическая настройка плотной компоновки 
            графика (да/нет, True/False). 
            Defaults to True.
        
        graph_size (tuple, optional):             
            размер графика в дюймах. 
            Defaults to (210/INCH, 297/INCH/2), 
                где константа INCH = 25.4 мм/дюйм.
                
        file_name (str, optional):                
            имя файла для сохранения на диске. 
            Defaults to None.
    
    Returns:
        None
        
    """
    
    # создание рисунка (Figure) и области рисования (Axes)
    fig, axes = plt.subplots(figsize=graph_size)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    # данные для построения графика
    if data_df_in is not None:
        data_df = data_df_in.copy()
        if x_ticklabels_list:
            data_df = data_df.set_index(pd.Index(x_ticklabels_list))
        if y_ticklabels_list:
            data_df.columns = y_ticklabels_list
    else:
        data_df = pd.DataFrame(data_XY_list_in)
        if x_ticklabels_list:
            data_df = data_df.set_index(pd.Index(x_ticklabels_list))
        if y_ticklabels_list:
            data_df.columns = y_ticklabels_list
        
    data_np = np.array(data_XY_list_in) if data_XY_list_in is not None \
        else np.array(data_df_in)
    
    data_df_rel = None
    
    if values_type == 'relative':         
        data_df_rel = data_df.copy()
        sum_data = np.sum(data_np)
        data_df.sum(axis=1)
        
        for col in data_df_rel.columns:
            data_df_rel[col] = data_df_rel[col] / sum_data
        
    # построение графика
    if values_type == "absolute":
        if not robust: 
            sns.heatmap(data_df.transpose(),
                #vmin=0, vmax=1,
                linewidth=0.5,
                cbar=True,
                fmt=fmt,
                annot=True,
                cmap=color_map,
                annot_kws = {"fontsize": annot_fontsize},
                ax=axes)
        else:
            sns.heatmap(data_df.transpose(),
                #vmin=0, vmax=1,
                linewidth=0.5,
                cbar=True,
                robust=True,
                fmt=fmt,
                annot=True,
                cmap=color_map,
                annot_kws = {"fontsize": annot_fontsize},
                ax=axes)
    else:
        if not robust: 
            sns.heatmap(data_df_rel.transpose(),
                vmin=0, vmax=1,
                linewidth=0.5,
                cbar=True,
                fmt=fmt,
                annot=True,
                cmap=color_map,
                annot_kws = {"fontsize": annot_fontsize},
                ax=axes)
        else:
            sns.heatmap(data_df_rel.transpose(),
                vmin=0, vmax=1,
                linewidth=0.5,
                cbar=True,
                robust=True,
                fmt=fmt,
                annot=True,
                cmap=color_map,
                annot_kws = {"fontsize": annot_fontsize},
                ax=axes)
    
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    axes.figure.axes[-1].yaxis.set_tick_params(labelsize = colorbar_fontsize)
            
    # автоматическая настройка плотной компоновки графика
    if tight_layout:
        fig.tight_layout()
        
    # вывод графика
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
    
    return


#------------------------------------------------------------------------------
#   Функция conjugacy_table_independence_check
#   
#   Проверка значимости связи между категориальными признаками 
#   в таблицах сопряженности
#------------------------------------------------------------------------------

def conjugacy_table_independence_check(    
    data,
    p_level:                         float = 0.95,
    methods:                         list = ['all'],
    correction_for_continuity_OR:    bool = False,
    logarithmic_method_OR:           bool = False,
    scale: str =                     'Rea&Parker',    
    expected_freq:                   bool = False):
    
    """
    Проверка значимости связи между категориальными
    признаками в таблицах сопряженности.
    
    Args:
        data (pd.core.frame.DataFrame, numpy.ndarray):    
            массив исходных данных.
            
        p_level (float, optional):                        
            доверительная вероятность. 
            Defaults to 0.95.
            
        methods (list, optional):                         
            список применяемых методов:  
                ['simple', 'odds ratio', 'chi2', 'chi2 coef', 'Fisher', 
                 'Barnard', 'Boschloo'] or ['all]. 
                Defaults to ['all]
                             
        correction_for_continuity_OR (bool, optional)     
            логический параметр: True/False - использовать/не использовать  
            для показателя odds ratio поправку на непрерывность 
            (наличие ячеек с нулевыми частотами).   
            Defaults to False.
            
        logarithmic_method_OR (bool, optional)            
            логический параметр: True/False - использовать/не использовать 
            для показателя odds ratio логарифмический метод при расчете 
            критических значений.     
            Defaults to False.
            
        scale (str, optional):                            
            вид шкалы для оценки тесноты связи 
            {'Cheddock', 'Evans', 'Rea&Parker'}.
            Defaults to 'Rea&Parker'.
            
        expected_freq (bool, optional):                   
            логический параметр: True/False - выводить/не выводить таблицу 
            теоретических частот.     
            Defaults to False.

    Returns:
        result_general (pd.core.frame.DataFrame):
            результат            
            
    """
    
    # особенность аргумента по умолчанию, который является списком (list)
    methods = methods or []    
    
    a_level = 1 - p_level   
    
    # исходные данные
    X = np.round(np.array(data))
           
    # параметры таблицы сопряженности
    N_rows = X.shape[0]
    N_cols = X.shape[1]
    N = X.size
    X_sum = np.sum(X)
    
    # фактические частоты для таблиц 2х2
    a = X[0][0]
    b = X[0][1]
    c = X[1][0]
    d = X[1][1]
    n = a + b + c + d   
    
    # ожидаемые частоты и предельные суммы
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.expected_freq.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.margins.html
    result_expected_freq = sci.stats.contingency.expected_freq(X)
    X_expected = np.array(result_expected_freq)
    
    # табл.значение квантиля норм.распр.
    u_p = sps.norm.ppf((1 + p_level)/2, 0, 1)
        
    # проверка условия размерности таблицы 2х2
    check_condition_2x2 = True if (N_rows == 2 and N_cols == 2) else False
    
    # проверка условий применимости критерия хи-квадрат
    # expected frequency less than 5 or 10
    check_condition_chi2_1 = True
    # proportion of cells with expected frequencies less than 5 is more than 20%
    check_condition_chi2_2 = True
    # sample size less than 30
    check_condition_chi2_3 = True
    # math domain error: expected frequency is 0
    check_condition_chi2_4 = True
    note_condition_chi2_1 = note_condition_chi2_2 = note_condition_chi2_3 = \
        note_condition_chi2_4 = ''
        
    if np.size(np.where(X.ravel() == 0)):    
            check_condition_chi2_4 = False
            note_condition_chi2_4 = 'expected frequency is 0 (math domain error)'
            
    if check_condition_2x2:
        # Функция np.ravel() преобразует матрицу в одномерный вектор
        if np.size(np.where(X_expected.ravel() < 5)):    
            check_condition_chi2_1 = False
            note_condition_chi2_1 = 'expected frequency less than 5'
    else:
        if np.size(np.where(X_expected.ravel() < 10)):
            check_condition_chi2_1 = False
            note_condition_chi2_1 = 'expected frequency less than 10'
    
    if not check_condition_2x2:
        # число ячеек с ожидаемым числом наблюдений менее 5
        N_expected_to_5 = np.count_nonzero(X_expected < 5)    
        if N_expected_to_5 / X_expected.size > 0.2:
            check_condition_chi2_2 = False
            note_condition_chi2_2 = \
                'proportion of cells with expected frequencies ' + \
                    'less than 5 is more than 20%'
                
    if X_sum < 30:
        check_condition_chi2_3 = False
        note_condition_chi2_3 = 'sample size less than 30'            
    
    # обработка ошибок
    try:
        if scale not in ['Cheddock', 'Evans', 'Rea&Parker']:
            note_error = \
                'Error! "scale" must be in ["Cheddock", "Evans", "Rea&Parker"]'
            raise ValueError(note_error)
    except ValueError as err:
        print(err)
    #    #(result, result_expected_freq) = (None, None)
    #    if not expected_freq:
    #        return None
    #    else:
    #        return (None, None)
        
    # оценка тесноты связи
    def scale_check(scale, coef_value):
        scale_func_dict = {
            'Cheddok': Cheddock_scale_check(coef_value),
            'Evans': Evans_scale_check(coef_value),
            "Rea&Parker": Rea_Parker_scale_check(coef_value)}
        result = scale_func_dict[scale] if (scale in scale_func_dict.keys()) \
            else '-'
        return result
    
    
    # BLOCK 0
    # -------
    
    # список для выдачи результата
    result_general = list()
    
    result_0 = pd.DataFrame({
        'p_level': p_level,
        'a_level': a_level,},
        index=[''])
    
    result_general.append(result_0)
    
    
    # BLOCK 1: simple methods
    # -----------------------
    
    if (('simple' in methods) or (methods == ['all'])) and check_condition_2x2:
        
        # коэффициент ассоциации Юла
        Q_calc = (a*d - b*c) / (a*d + b*c)
        D_Q = 1/4 * (1-Q_calc**2)**2 * (1/a + 1/b + 1/c + 1/d)
        ASE_Q = sqrt(D_Q)
        Z_Q = abs(Q_calc) / ASE_Q
        Q_crit = u_p * ASE_Q
        a_calc_Q = (1 - sci.stats.norm.cdf(Z_Q, loc=0, scale=1))*2
        conclusion_Q = 'significant' if abs(Q_calc) >= Q_crit \
            else 'not significant'        
    
        # коэффициент коллигации Юла
        Y_calc = (sqrt(a*d) - sqrt(b*c)) / (sqrt(a*d) + sqrt(b*c))
        D_Y = 1/16 * (1-Y_calc**2)**2 * (1/a + 1/b + 1/c + 1/d)
        ASE_Y = sqrt(D_Y)
        Z_Y = abs(Y_calc) / ASE_Y
        Y_crit = u_p * ASE_Y
        a_calc_Y = (1 - sci.stats.norm.cdf(Z_Y, loc=0, scale=1))*2
        conclusion_Y = 'significant' if abs(Y_calc) >= Y_crit \
            else 'not significant'
    
        # коэффициент контингенции Пирсона
        V_calc = (a*d - b*c) / sqrt((a + b)*(a + c)*(b + d)*(c + d))
        D_V = 1/n * (1-V_calc**2) + \
            1/n * (V_calc + 1/2 * V_calc**2) * ((a-d)**2 - \
                (b-c)**2)/sqrt((a+b)*(a+c)*(b+d)*(c+d)) - \
                    3/(4*n)*V_calc**2 * (((a+b-c-d)**2 / ((a+b)*(c+d))) \
                                         - ((a+c-b-d)**2 / ((a+c)*(b+d))))
        ASE_V = sqrt(D_V)
        Z_V = abs(V_calc) / ASE_V
        V_crit = u_p * sqrt(D_V)
        a_calc_V = (1 - sci.stats.norm.cdf(Z_V, loc=0, scale=1))*2
        conclusion_V = 'significant' if abs(V_calc) >= V_crit \
            else 'not significant'
    
        # сводка результатов
        notation_list = ('Q', 'Y', 'V')
        coef_value_list = np.array([Q_calc, Y_calc, V_calc])
        ASE_list = np.array([ASE_Q, ASE_Y, ASE_V])
        critical_value_list = (Q_crit, Y_crit, V_crit)
        a_calc_list = (a_calc_Q, a_calc_Y, a_calc_V)
        significance_check_list = (conclusion_Q, conclusion_Y, conclusion_V)
    
        # доверительные интервалы
        #func_of_significance = lambda value, significance_check: value if significance_check == 'significant' else '-'
        conf_int_low_list = coef_value_list - ASE_list*u_p
        conf_int_low_high = coef_value_list + ASE_list*u_p
    
        # оценка тесноты связи
        if scale=='Evans':
            scale_list = np.vectorize(Evans_scale_check)(coef_value_list, \
                                                         notation_list)
        elif scale=='Cheddock':
            scale_list = np.vectorize(Cheddock_scale_check)(coef_value_list, \
                                                            notation_list)
        elif scale=='Rea&Parker':
            scale_list = np.vectorize(Rea_Parker_scale_check)(coef_value_list,\
                                                              notation_list)
        else:
            scale_list = np.vectorize(Rea_Parker_scale_check)(coef_value_list,\
                                                              ('error'))
        
        # Создадим DataFrame для сводки результатов
        scale_name = f'{scale} scale'
        result_1 = pd.DataFrame({
            'name': (
                'Yule’s Coefficient of Association',
                'Yule’s Coefficient of Colligation',
                'Pearson’s contingency coefficient'),
            'notation': notation_list,
            #'p_level': (p_level),
            #'a_level': (a_level),
            'coef_value': coef_value_list,
            'ASE': ASE_list,
            'crit_value': critical_value_list,
            '|coef_value| >= crit_value': (abs(coef_value_list) >= \
                                           critical_value_list),
            'a_calc': a_calc_list,
            'a_calc <= a_level': (elem <= a_level for elem in a_calc_list),
            'significance_check': significance_check_list,
            'conf_int_low': conf_int_low_list,
            'conf_int_high': conf_int_low_high,
            scale_name: scale_list
            }) 
        
        result_general.append(result_1)
        
    else:
        if not check_condition_2x2:
            note = "Yule’s Coefficient of Association (Q), " + \
                "Yule’s Coefficient of Colligation (Y) and " + \
                "Pearson’s contingency coefficient (V) " + \
                "cannot be calculated: the size of the table is not 2*2"
            print(note)
            
            
    # BLOCK 2: odds ratio
    # -------------------
    
    if (('odds ratio' in methods) \
        or (methods == ['all'])) and check_condition_2x2:
        
        # отношение шансов (odds ratio)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.odds_ratio.html
        res_OR = sps.contingency.odds_ratio(X)
        OR_calc = res_OR.statistic
        OR_ci = res_OR.confidence_interval(confidence_level = p_level)
        #print(OR_ci)
        
        #OR_calc = (a/b)/(c/d) if not correction_for_continuity_OR else ((a+0.5)/(b+0.5))/((c+0.5)/(d+0.5))
        D_OR = (OR_calc**2) * (1/a + 1/b + 1/c + 1/d) \
            if not correction_for_continuity_OR \
                else (OR_calc**2) * \
                    (1/(a+0.5) + 1/(b+0.5) + 1/(c+0.5) + 1/(d+0.5))
        ASE_OR = np.sqrt(D_OR)
        
        if logarithmic_method_OR:
            D_lnOR = (1/a + 1/b + 1/c + 1/d) \
                if not correction_for_continuity_OR \
                    else (1/(a+0.5) + 1/(b+0.5) + 1/(c+0.5) + 1/(d+0.5))
            ASE_lnOR = np.sqrt(D_lnOR)
            Z_lnOR = abs(np.log(OR_calc) / ASE_lnOR)
            lnOR_crit = u_p * sqrt(D_lnOR)
            a_calc_lnOR = (1 - sci.stats.norm.cdf(Z_lnOR, loc=0, scale=1))*2
            conclusion_lnOR = 'significant' \
                if abs(np.log(OR_calc)) >= lnOR_crit else 'not significant'
        else:
            Z_OR = abs(OR_calc / ASE_OR)
            OR_crit = u_p * sqrt(D_OR)
            a_calc_OR = (1 - sci.stats.norm.cdf(Z_OR, loc=0, scale=1))*2
            conclusion_OR = 'significant' \
                if OR_calc >= OR_crit else 'not significant'
        
        # оценка связи - отношение шансов (odds ratio)
        connection_type_OR = 'negative connection' \
            if OR_calc < 1 else 'positive connection' \
                if OR_calc > 1 else 'no connection'
        
        # сводка результатов
        notation_list = ('OR')
        coef_value_list = (OR_calc)
        if logarithmic_method_OR:
            ln_coef_value_list = (np.log(OR_calc))
            ASE_ln_list = (ASE_lnOR)            
            critical_ln_value_list = (lnOR_crit)
            a_calc_ln_list = (a_calc_lnOR)
        else:
            ASE_list = (ASE_OR)
            critical_value_list = (OR_crit)
            a_calc_list = (a_calc_OR)
        significance_check_list = (conclusion_OR) \
            if not logarithmic_method_OR else (conclusion_lnOR)
    
        # доверительные интервалы
        conf_int_low_list = (OR_ci.low)
        conf_int_high_list = (OR_ci.high)
        
        '''if logarithmic_method_OR:
            conf_int_low_list = (np.exp(np.log(OR_calc) - ASE_lnOR*u_p), '-')
            conf_int_high_list = (np.exp(np.log(OR_calc) + ASE_lnOR*u_p), '-')
        else:
            conf_int_low_list = (OR_calc - ASE_OR*u_p, '-')
            conf_int_high_list = (OR_calc + ASE_OR*u_p, '-')'''
                
        # Создадим DataFrame для сводки результатов
        if not logarithmic_method_OR:
            result_2 = pd.DataFrame({
                'name': ('Odds ratio'),
                'notation':                            notation_list,
                #'p_level':                            (p_level),
                #'a_level':                            (a_level),
                'calc_value':                          coef_value_list,
                'connection_type':                     (connection_type_OR),
                'ASE':                                 ASE_list,
                'crit_value':                          critical_value_list,
                '|calc_value| >= crit_value':          (OR_calc >= OR_crit),
                'a_calc':                              a_calc_list,
                'a_calc <= a_level':                   (a_calc_list <= a_level),
                'significance_check':                  significance_check_list,
                'conf_int_low':                        conf_int_low_list,
                'conf_int_high':                       conf_int_high_list
                },
                index=[0])
            
        else:
            result_2 = pd.DataFrame({
                'name': ('Odds ratio'),
                'notation':                                notation_list,
                #'p_level':                                 (p_level),
                #'a_level':                                 (a_level),
                'calc_value':                              coef_value_list,
                'connection_type':                         (connection_type_OR),
                'ASE(ln(calc_value))':                     ASE_ln_list,
                'crit_ln(calc_value)':\
                     critical_ln_value_list,
                '|ln(calc_value)| >= crit_ln(calc_value)': \
                    (np.log(OR_calc) >= lnOR_crit),
                'a_calc':                                  a_calc_ln_list,
                'a_calc <= a_level':\
                       (a_calc_ln_list <= a_level),
                'significance_check':\
                      significance_check_list,
                'conf_int_low':                            conf_int_low_list,
                'conf_int_high':                           conf_int_high_list
                },
                index=[0])
    
        result_general.append(result_2)            
        
    else:
        if not check_condition_2x2:
            note = 'Odds ratio (OR) and relative risk (RR) ' + \
                'cannot be calculated: the size of the table is not 2*2'
            print(note)
    
    
    # BLOCK 3: chi2 test
    # ------------------
    
    if (('chi2' in methods) or (methods == ['all'])) and \
        (check_condition_chi2_1 and check_condition_chi2_2 \
         and check_condition_chi2_3 and check_condition_chi2_4):
        
        # критерий хи-квадрат (без поправки Йетса)
        # https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
        s_calc_chi2 = '-'
        critical_value_chi2 = '-'
        independence_check_chi2_s = '-'
        a_calc_chi2 = '-'
        independence_check_chi2_a = '-'
        
        if not check_condition_2x2:
            (s_calc_chi2, p_chi2, dof_chi2, ex_chi2) = \
                sci.stats.chi2_contingency(X, correction=False)
            critical_value_chi2 = sci.stats.chi2.ppf(p_level, dof_chi2)
            independence_check_chi2_s = s_calc_chi2 <= critical_value_chi2
            a_calc_chi2 = p_chi2
            # альтернативный расчет: a_calc_chi2 = 1 - sci.stats.chi2.cdf(s_calc_chi2, dof_chi2)
            independence_check_chi2_a = a_calc_chi2 > a_level
            conclusion_chi2 = 'categories are independent' if \
                independence_check_chi2_s else 'categories are not independent'
        else:
            conclusion_chi2 = '2x2'
            
        # критерий хи-квадрат (с поправкой Йетса) (только для таблиц 2х2)
        # https://en.wikipedia.org/wiki/Yates%27s_correction_for_continuity
        # https://ru.wikipedia.org/wiki/%D0%9F%D0%BE%D0%BF%D1%80%D0%B0%D0%B2%D0%BA%D0%B0_%D0%99%D0%B5%D0%B9%D1%82%D1%81%D0%B0
        s_calc_chi2_Yates = '-'
        critical_value_chi2_Yates = '-'
        independence_check_chi2_s_Yates = '-'
        a_calc_chi2_Yates = '-'
        independence_check_chi2_a_Yates = '-'
        
        if check_condition_2x2:
            (s_calc_chi2_Yates, p_chi2_Yates, dof_chi2_Yates, ex_chi2_Yates) =\
                sci.stats.chi2_contingency(X, correction=True)
            critical_value_chi2_Yates = \
                sci.stats.chi2.ppf(p_level, dof_chi2_Yates)
            independence_check_chi2_s_Yates = \
                s_calc_chi2_Yates <= critical_value_chi2_Yates
            a_calc_chi2_Yates = p_chi2_Yates
            independence_check_chi2_a_Yates = a_calc_chi2_Yates > a_level
            conclusion_chi2_Yates = 'categories are independent' \
                if independence_check_chi2_s_Yates \
                    else 'categories are not independent'
        else:
            conclusion_chi2_Yates = 'not 2x2'
            
        # критерий хи-квадрат (с поправкой на правдоподобие)
        s_calc_chi2_likelihood = '-'
        critical_value_chi2_likelihood = '-'
        independence_check_chi2_s_likelihood = '-'
        a_calc_chi2_likelihood = '-'
        independence_check_chi2_a_likelihood = '-'
        
        if check_condition_2x2:
            E = np.array(sci.stats.chi2_contingency(X, correction=True)[3])
        else:
            E = np.array(sci.stats.chi2_contingency(X, correction=False)[3])
        s_calc_chi2_likelihood = 2*np.sum([[X[i,j]*log(X[i,j]/E[i,j]) \
            for j in range(N_cols)] for i in range(N_rows)])
        df = (N_rows-1)*(N_cols-1)    
        critical_value_chi2_likelihood = sci.stats.chi2.ppf(p_level, df)
        independence_check_chi2_s_likelihood = \
            s_calc_chi2_likelihood <= critical_value_chi2_likelihood
        a_calc_chi2_likelihood = 1 - sci.stats.chi2.cdf(s_calc_chi2_likelihood,\
                                                        df, loc=0, scale=1)
        independence_check_chi2_a_likelihood = a_calc_chi2_likelihood > a_level
        conclusion_chi2_likelihood = 'categories are independent' \
            if independence_check_chi2_s_likelihood \
                else 'categories are not independent'
            
            
        # заполним DataFrame для сводки результатов тестов
        result_3 = pd.DataFrame({
            'test': (
                'Chi-squared test',
                "Chi-squared test (with Yates's correction for 2x2)",
                "Likelihood-corrected Chi-squared test"),
            #'p_level': (p_level),
            # #'a_level': (a_level),
            'statistic': [s_calc_chi2, s_calc_chi2_Yates, \
                          s_calc_chi2_likelihood],
            'critical_value': [critical_value_chi2, critical_value_chi2_Yates, \
                               critical_value_chi2_likelihood],
            'statistic <= critical_value': [independence_check_chi2_s, \
                independence_check_chi2_s_Yates, \
                    independence_check_chi2_s_likelihood],
            'a_calc': [a_calc_chi2, a_calc_chi2_Yates, a_calc_chi2_likelihood],
            'a_calc >= a_level': [independence_check_chi2_a, \
                independence_check_chi2_a_Yates, \
                    independence_check_chi2_a_likelihood],
            'conclusion': [conclusion_chi2, conclusion_chi2_Yates, \
                           conclusion_chi2_likelihood]
            })   
        
        result_general.append(result_3)                 
        
    else:
        if not (check_condition_chi2_1 and check_condition_chi2_2 and \
                check_condition_chi2_3 and check_condition_chi2_4):
            #note = 'The chi-square criterion is not applicable: ' + \
            #    f'{note_condition_chi2_1}, {note_condition_chi2_2}, ' + \
            #        f'{note_condition_chi2_3}, {note_condition_chi2_4}'
            note = 'The chi-square criterion is not applicable: ' + \
                (f'{note_condition_chi2_1}' if not check_condition_chi2_1 \
                    else '') + \
                (f'{note_condition_chi2_2}' if not check_condition_chi2_2 \
                    else '') + \
                (f'{note_condition_chi2_3}' if not check_condition_chi2_3 \
                    else '') + \
                (f'{note_condition_chi2_4}' if not check_condition_chi2_4 \
                    else '')
                            
            print(note)
        
        
    # BLOCK 4: chi2 coefficients
    # --------------------------    
    
    if (('chi2 coef' in methods) or (methods == ['all'])) and \
        (check_condition_chi2_1 and check_condition_chi2_2 \
            and check_condition_chi2_3 and check_condition_chi2_4):
        
        # меры связи, основанные на критерии хи-квадрат и их стандартные ошибки (см. [Кендалл, с.749])
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html
        
        if check_condition_2x2:
            D_chi2 = 4*s_calc_chi2_Yates
            # коэффициент сопряженности Пирсона
            coef_value_C = sci.stats.contingency.association(X, \
                method='pearson', correction='True')
            ASE_C = N**2 / (4*s_calc_chi2_Yates*(X_sum + \
                s_calc_chi2_Yates)**3)*D_chi2
            # коэффициент ассоциации фи
            coef_value_Phi = sqrt(sci.stats.chi2_contingency(X, \
                correction=True)[0] / X.sum())
            # коэффициент сопряженности Крамера
            coef_value_V = sci.stats.contingency.association(X, \
                method='cramer', correction='True')
            ASE_V = np.sqrt(1 / (X_sum * np.min([N_rows-1, N_cols-1])))
            # коэффициент сопряженности Чупрова
            coef_value_T = sci.stats.contingency.association(X, \
                method='tschuprow', correction='True')
            ASE_T = np.sqrt(1 / (X_sum * np.sqrt((N_rows-1)*(N_cols-1))))
        else:
            D_chi2 = 4*s_calc_chi2
            # коэффициент сопряженности Пирсона
            coef_value_C = sci.stats.contingency.association(X, \
                method='pearson', correction='False')
            ASE_C = np.sqrt(X_sum**2 / (4*s_calc_chi2*(X_sum + \
                                                       s_calc_chi2)**3)*D_chi2)
            # коэффициент ассоциации фи
            coef_value_Phi = sqrt(sci.stats.chi2_contingency(X, \
                correction=False)[0] / X.sum())
            # коэффициент сопряженности Крамера
            coef_value_V = sci.stats.contingency.association(X, \
                method='cramer', correction='False')
            ASE_V = np.sqrt(1 / (X_sum * np.min([N_rows-1, N_cols-1])))
            # коэффициент сопряженности Чупрова
            coef_value_T = sci.stats.contingency.association(X, method='tschuprow', correction='False')
            ASE_T = np.sqrt(1 / (X_sum * np.sqrt((N_rows-1)*(N_cols-1))))
        
        # заполним DataFrame для сводки результатов коэффициентов
        coef_value_list = (coef_value_C, coef_value_Phi, coef_value_V, \
                           coef_value_T)
        scale_name = f'{scale} scale'
    
        result_4 = pd.DataFrame({
            'name': (
                "Pearson's C",
                "Phi coefficient",
                "Cramér's V",
                "Tschuprow's T"),
            'notation': ('С', 'φ', 'V', 'T'),
            #'p_level': (p_level),
            #'a_level': (a_level),
            'coef_value': coef_value_list,
            'ASE': (ASE_C, '-', ASE_V, ASE_T),
            #'crit_value': (''),
            # '|coef_value| >= crit_value': (''),
            # 'significance_check': (''),        
            # 'conf_int_low': (confidence_interval_min),
            # 'conf_int_high': (confidence_interval_max),
            scale_name: list(map(scale_check, \
                [scale for i in range(len(coef_value_list))], coef_value_list))
            })
        
        result_general.append(result_4)                 
    
    
    # BLOCK 5: Fisher's exact test
    # ----------------------------  
        
    if (('Fisher' in methods) or (methods == ['all'])) and check_condition_2x2:
        
        # точный критерий Фишера (Fisher's exact test)
        # https://en.wikipedia.org/wiki/Fisher's_exact_test
        # https://ru.wikipedia.org/wiki/%D0%A2%D0%BE%D1%87%D0%BD%D1%8B%D0%B9_%D1%82%D0%B5%D1%81%D1%82_%D0%A4%D0%B8%D1%88%D0%B5%D1%80%D0%B0
        # https://medstatistic.ru/methods/methods5.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
        (odds_ratio, a_Fisher_calc_two_sided) = \
            sci.stats.fisher_exact(X, alternative='two-sided')
        Fisher_calc_alternative = 'greater' if odds_ratio > 1 else 'less'
        a_Fisher_calc_one_sided = sci.stats.fisher_exact(X, \
            alternative=Fisher_calc_alternative)[1]
        independence_check_Fisher_one_sided = a_Fisher_calc_one_sided > a_level
        conclusion_Fisher_one_sided = 'categories are independent' \
            if independence_check_Fisher_one_sided \
                else 'categories are not independent'
        independence_check_Fisher_two_sided = a_Fisher_calc_two_sided > a_level
        conclusion_Fisher_two_sided = 'categories are independent' \
            if independence_check_Fisher_two_sided \
                else 'categories are not independent'
        
        # заполним DataFrame для сводки результатов тестов
        result_5 = pd.DataFrame({
            'test': ("Fisher's exact test (one-sided)", "Fisher's exact test (two-sided)"),
            #'p_level': (p_level),
            # #'a_level': (a_level),
            'a_calc': [a_Fisher_calc_one_sided, a_Fisher_calc_two_sided],
            'a_calc >= a_level': [independence_check_Fisher_one_sided, \
                                  independence_check_Fisher_two_sided],
            'conclusion': [conclusion_Fisher_one_sided, \
                           conclusion_Fisher_two_sided]
            })
        
        result_general.append(result_5)                 
        
    else:
        if not check_condition_2x2:
            note = "The Fisher's exact test cannot be used: " + \
                "the size of the table is not 2*2" 
            print(note)
    
    
    # BLOCK 6: Barnard's exact test
    # -----------------------------
    
    if (('Barnard' in methods) or (methods == ['all'])) and check_condition_2x2:
        
        # тест Барнарда 
        # https://en.wikipedia.org/wiki/Barnard's_test
        # https://wikidea.ru/wiki/Barnard%27s_test
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.barnard_exact.html
        res = sci.stats.barnard_exact(X, alternative='two-sided')
        (s_calc_Barnard, a_calc_Barnard) = (res.statistic, res.pvalue)
        independence_check_Barnard = a_calc_Barnard > a_level
        conclusion_Barnard = 'categories are independent' \
            if independence_check_Barnard else 'categories are not independent'
        
        # заполним DataFrame для сводки результатов тестов
        result_6 = pd.DataFrame({
            'test': ("Barnard's exact test"),
            #'p_level': (p_level),
            #'a_level': (a_level),
            'statistic': [s_calc_Barnard],
            #'critical_value': [],
            #'statistic <= critical_value': [],
            'a_calc': [a_calc_Barnard],
            'a_calc >= a_level': [independence_check_Barnard],
            'conclusion': [conclusion_Barnard]
            })  
        
        result_general.append(result_6)                 
    
    else:
        if not check_condition_2x2:
            note = "The Barnard's test cannot be used: " + \
                "the size of the table is not 2*2"
            print(note)
    
    
    # BLOCK 7: Boschloo's_exact test
    # ------------------------------
    
    if (('Boschloo' in methods) \
        or (methods == ['all'])) and check_condition_2x2:
        
        # тест Бошлу 
        # # https://en.wikipedia.org/wiki/Boschloo%27s_test
        # # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boschloo_exact.html             
        res = sci.stats.boschloo_exact(X, alternative='two-sided')
        (s_calc_Boschloo, a_calc_Boschloo) = (res.statistic, res.pvalue)
        independence_check_Boschloo = a_calc_Boschloo > a_level
        conclusion_Boschloo = 'categories are independent' \
            if independence_check_Boschloo \
                else 'categories are not independent'
        
        # заполним DataFrame для сводки результатов тестов
        result_7 = pd.DataFrame({
            'test': ("Boschloo's exact test"),
            #'p_level': (p_level),
            #'a_level': (a_level),
            'statistic': [s_calc_Boschloo],
            #'critical_value': [],
            #'statistic <= critical_value': [],
            'a_calc': [a_calc_Boschloo],
            'a_calc >= a_level': [independence_check_Boschloo],
            'conclusion': [conclusion_Boschloo]
            })  
        
        result_general.append(result_7)                 
    
    else:
        if not check_condition_2x2:
            note = "The Boschloo's test cannot be used: " + \
                "the size of the table is not 2*2"
            print(note)
    
        
    # EXPECTED FREQ
    # -------------
    
    if expected_freq:
        if isinstance(data, pd.core.frame.DataFrame):
            result_expected_freq_df = pd.DataFrame(
                result_expected_freq,
                index = data.index,
                columns = pd.MultiIndex.from_product([['expected frequencies'],\
                                                      data.columns]))
        else: 
            result_expected_freq_df = pd.DataFrame(
                result_expected_freq,
                columns = pd.MultiIndex.from_product([['expected frequencies'],\
                                                      range(0, N_cols)]))
        
        result_general.append(result_expected_freq_df)     
    
    # RETURN RESULTS
    # --------------
        
    return result_general





#==============================================================================
#               9. ФУНКЦИИ ДЛЯ КОРРЕЛЯЦИОННОГО АНАЛИЗА
#==============================================================================


#------------------------------------------------------------------------------
#   Функция Cheddock_scale_check
#   Оценка тесноты связи по шкале Чеддока
#------------------------------------------------------------------------------

def Cheddock_scale_check(r, name: str='r'):
    
    """
    Функция для оценки тесноты связи по шкале Чеддока

    Args:
        r:                       Коэффициент, характеризующий тесноту связи
        name (str, optional):    Обозначение коэффициента
                                 Defaults to 'r'

    Returns:
        conclusion_Cheddock_scale (str):    результат    
                             
    """
    
    # задаем шкалу Чеддока
    Cheddock_scale = {
        f'no correlation (|{name}| <= 0.1)':    0.1,
        f'very weak (0.1 < |{name}| <= 0.2)':   0.2,
        f'weak (0.2 < |{name}| <= 0.3)':        0.3,
        f'moderate (0.3 < |{name}| <= 0.5)':    0.5,
        f'perceptible (0.5 < |{name}| <= 0.7)': 0.7,
        f'high (0.7 < |{name}| <= 0.9)':        0.9,
        f'very high (0.9 < |{name}| <= 0.99)':  0.99,
        f'functional (|{name}| > 0.99)':        1.0}
    
    r_scale = list(Cheddock_scale.values())
    for i, elem in enumerate(r_scale):
        if abs(r) <= elem:
            conclusion_Cheddock_scale = list(Cheddock_scale.keys())[i]
            break
    return conclusion_Cheddock_scale  



#------------------------------------------------------------------------------
#   Функция Evans_scale_check
#   Оценка тесноты связи по шкале Эванса
#------------------------------------------------------------------------------

def Evans_scale_check(r, name: str='r'):
    
    """
    Функция для оценки тесноты связи по шкале Эванса

    Args:
        r:                       Коэффициент, характеризующий тесноту связи
        name (str, optional):    Обозначение коэффициента
                                 Defaults to 'r'
    
    Returns:
        conclusion_Evans_scale (str):    результат        
                         
    """
    
    # задаем шкалу Эванса
    Evans_scale = {
        f'very weak (|{name}| < 0.19)':         0.2,
        f'weak (0.2 < |{name}| <= 0.39)':       0.4,  
        f'moderate (0.4 < |{name}| <= 0.59)':   0.6,
        f'strong (0.6 < |{name}| <= 0.79)':     0.8,
        f'very strong (0.8 < |{name}| <= 1.0)': 1.0}
    
    r_scale = list(Evans_scale.values())
    for i, elem in enumerate(r_scale):
        if abs(r) <= elem:
            conclusion_Evans_scale = list(Evans_scale.keys())[i]
            break
    return conclusion_Evans_scale  



#------------------------------------------------------------------------------
#   Функция Rea_Parker_scale_check
#   Оценка тесноты связи по шкале Rea&Parker
#------------------------------------------------------------------------------

def Rea_Parker_scale_check(r, name: str='r'):
    
    """
    Функция для оценки тесноты связи по шкале Rea&Parker

    Args:
        r:                       Коэффициент, характеризующий тесноту связи
        name (str, optional):    Обозначение коэффициента
                                 Defaults to 'r'
    
    Returns:
        conclusion_Rea_Parker (str):    результат    
                             
    """
    
    # задаем шкалу Rea_Parker
    Rea_Parker_scale = {
        f'unessential (|{name}| < 0.1)':                  0.1,
        f'weak (0.1 <= |{name}| < 0.2)':                  0.2,  
        f'middle (0.2 <= |{name}| < 0.4)':                0.4,
        f'relatively strong (0.4 <= |{name}| < 0.6)':     0.6,
        f'strong (0.6 <= |{name}| < 0.8)':                0.8,
        f'very strong (0.8 <= |{name}| <= 1.0)':          1.0}
    
    r_scale = list(Rea_Parker_scale.values())
    for i, elem in enumerate(r_scale):
        if abs(r) <= elem:
            conclusion_Rea_Parker_scale = list(Rea_Parker_scale.keys())[i]
            break
    return conclusion_Rea_Parker_scale



#------------------------------------------------------------------------------
#   Функция corr_coef_check
#
#   Функция для расчета и анализа коэффициента корреляции Пирсона
#------------------------------------------------------------------------------

def corr_coef_check(
        X, Y,
        p_level:    float = 0.95, 
        scale:      str = 'Cheddok'):
            
    """
    Расчет и проверка значимости коэффициента линейной корреляции Пирсона.
        
    Args:
        X, Y:                      
            массивы исходных данных.
            
        p_level (float, optional):    
            доверительная вероятность. 
            Defaults to 0.95.

        scale (str, optional):               
            шкала для оценки тесноты связи.
            Defaults to 'Evans'.
            
    Returns:
        result (pd.core.frame.DataFrame):
            результат.
    
    """
    
    a_level = 1 - p_level
    X = np.array(X)
    Y = np.array(Y)
    n_X = len(X)
    n_Y = len(Y)
    # оценка коэффициента линейной корреляции средствами scipy
    corr_coef, a_corr_coef_calc = sci.stats.pearsonr(X, Y)
    # несмещенная оценка коэффициента линейной корреляции (при n < 15)
    # (см.Кобзарь, с.607)
    if n_X < 15:
        corr_coef = corr_coef * (1 + (1 - corr_coef**2) / (2*(n_X-3)))
    # проверка гипотезы о значимости коэффициента корреляции
    t_corr_coef_calc = abs(corr_coef) * sqrt(n_X-2) / sqrt(1 - corr_coef**2)
    t_corr_coef_table = sci.stats.t.ppf((1 + p_level)/2 , n_X - 2)
    conclusion_corr_coef_sign = 'significance' if \
        t_corr_coef_calc >= t_corr_coef_table else 'not significance'
    # доверительный интервал коэффициента корреляции
    if t_corr_coef_calc >= t_corr_coef_table:
        z1 = np.arctanh(corr_coef) - \
            sci.stats.norm.ppf((1 + p_level)/2, 0, 1) / sqrt(n_X-3) - \
                corr_coef / (2*(n_X-1))
        z2 = np.arctanh(corr_coef) + \
            sci.stats.norm.ppf((1 + p_level)/2, 0, 1) / sqrt(n_X-3) - \
                corr_coef / (2*(n_X-1))
        corr_coef_conf_int_low = tanh(z1)
        corr_coef_conf_int_high = tanh(z2)
    else:
        corr_coef_conf_int_low = corr_coef_conf_int_high = '-'    
    # оценка тесноты связи
    if scale=='Cheddok':
        conclusion_corr_coef_scale = scale + ': ' + \
            Cheddock_scale_check(corr_coef)
    elif scale=='Evans':
        conclusion_corr_coef_scale = scale + ': ' + \
            Evans_scale_check(corr_coef)
    # формируем результат            
    result = pd.DataFrame({
        'notation': ('r'),
        'coef_value': (corr_coef),
        'coef_value_squared': (corr_coef**2),
        'p_level': (p_level),
        'a_level': (a_level),
        't_calc': (t_corr_coef_calc),
        't_table': (t_corr_coef_table),
        't_calc >= t_table': (t_corr_coef_calc >= t_corr_coef_table),
        'a_calc': (a_corr_coef_calc),
        'a_calc <= a_level': (a_corr_coef_calc <= a_level),
        'significance_check': (conclusion_corr_coef_sign),
        'conf_int_low': (corr_coef_conf_int_low),
        'conf_int_high': (corr_coef_conf_int_high),
        'scale': (conclusion_corr_coef_scale)
        },
        index=['Correlation coef.'])
        
    return result        


#------------------------------------------------------------------------------
#   Функция corr_ratio_check
#
#   Функция для расчета и анализа корреляционного отношения
#------------------------------------------------------------------------------

def corr_ratio_check(
        X, Y, 
        p_level:        float = 0.95, 
        orientation:    str = 'XY', 
        scale:          str = 'Cheddok'):
    
    """
    Расчет и проверка значимости корреляционного отношения.
        
    Args:
        X, Y:                      
            массивы исходных данных.
            
        p_level (float, optional):    
            доверительная вероятность. 
            Defaults to 0.95.

        orientation (str, optional):               
            направление связи ('XY' или 'YX').
            Defaults to 'XY'.
            
        scale (str, optional):               
            шкала для оценки тесноты связи.
            Defaults to 'Evans'.
            
    Returns:
        result (pd.core.frame.DataFrame):
            результат.
    
    Note:
        при значениях η близких к 0 или 1 левая или правая граница 
        доверительного интервала может выходить за пределы отрезка [0; 1], 
        теряя содержательный смысл (Айвазян С.А. и др. Прикладная статистика: 
        исследование зависимостей. - М.: Финансы и статистика, 1985. - с.80]). 
        Причина этого - в аппроксимационном подходе к определению границ 
        доверительного интервала. Подобные нежелательные явления возможны, 
        и их нужно учитывать при выполнении анализа.            
    
    """
    
    a_level = 1 - p_level
    X = np.array(X)
    Y = np.array(Y)
    n_X = len(X)
    n_Y = len(Y)    
    # запишем данные в DataFrame
    matrix_XY_df = pd.DataFrame({
        'X': X,
        'Y': Y})
    # число интервалов группировки
    group_int_number = lambda n: round (3.31*log(n, 10)+1) if \
        round (3.31*log(n, 10)+1) >=2 else 2
    K_X = group_int_number(n_X)
    K_Y = group_int_number(n_Y)
    # группировка данных и формирование корреляционной таблицы
    cut_X = pd.cut(X, bins=K_X)
    cut_Y = pd.cut(Y, bins=K_Y)
    matrix_XY_df['cut_X'] = cut_X
    matrix_XY_df['cut_Y'] = cut_Y
    CorrTable_df = pd.crosstab(
        index=matrix_XY_df['cut_X'],
        columns=matrix_XY_df['cut_Y'],
        rownames=['cut_X'],
        colnames=['cut_Y'],
        dropna=False)
    CorrTable_np = np.array(CorrTable_df)
    # итоги корреляционной таблицы по строкам и столбцам
    n_group_X = [np.sum(CorrTable_np[i]) for i in range(K_X)]
    n_group_Y = [np.sum(CorrTable_np[:,j]) for j in range(K_Y)]
    # среднегрупповые значения переменной X
    Xboun_mean = \
        [(CorrTable_df.index[i].left + CorrTable_df.index[i].right)/2 \
            for i in range(K_X)]
    # исправляем значения в крайних интервалах
    Xboun_mean[0] = (np.min(X) + CorrTable_df.index[0].right)/2    
    Xboun_mean[K_X-1] = (CorrTable_df.index[K_X-1].left + np.max(X))/2
    # среднегрупповые значения переменной Y
    Yboun_mean = \
        [(CorrTable_df.columns[j].left + CorrTable_df.columns[j].right)/2 \
            for j in range(K_Y)]
    # исправляем значения в крайних интервалах
    Yboun_mean[0] = (np.min(Y) + CorrTable_df.columns[0].right)/2    
    Yboun_mean[K_Y-1] = (CorrTable_df.columns[K_Y-1].left + np.max(Y))/2
    # средневзевешенные значения X и Y для каждой группы
    Xmean_group = \
        [np.sum(CorrTable_np[:,j] * Xboun_mean) / n_group_Y[j] \
            for j in range(K_Y)]
    for i, elem in enumerate(Xmean_group):
        if isnan(elem):
            Xmean_group[i] = 0
    Ymean_group = \
        [np.sum(CorrTable_np[i] * Yboun_mean) / n_group_X[i] \
            for i in range(K_X)]
    for i, elem in enumerate(Ymean_group):
        if isnan(elem):
            Ymean_group[i] = 0
    # общая дисперсия X и Y
    Sum2_total_X = np.sum(n_group_X * (Xboun_mean - np.mean(X))**2)
    Sum2_total_Y = np.sum(n_group_Y * (Yboun_mean - np.mean(Y))**2)
    # межгрупповая дисперсия X и Y (дисперсия групповых средних)
    Sum2_between_group_X = np.sum(n_group_Y * (Xmean_group - np.mean(X))**2)
    Sum2_between_group_Y = np.sum(n_group_X * (Ymean_group - np.mean(Y))**2)
    # эмпирическое корреляционное отношение
    corr_ratio_XY = sqrt(Sum2_between_group_Y / Sum2_total_Y)
    corr_ratio_YX = sqrt(Sum2_between_group_X / Sum2_total_X)
    try:
        if orientation!='XY' and orientation!='YX':
            raise ValueError("Error! Incorrect orientation!")
        if orientation=='XY':
            corr_ratio = corr_ratio_XY
        elif orientation=='YX':
            corr_ratio = corr_ratio_YX
    except ValueError as err:
        print(err)
    # проверка гипотезы о значимости корреляционного отношения
    F_corr_ratio_calc = \
        (n_X - K_X)/(K_X - 1) * corr_ratio**2 / (1 - corr_ratio**2)
    dfn = K_X - 1
    dfd = n_X - K_X
    F_corr_ratio_table = \
        sci.stats.f.ppf(p_level, dfn, dfd, loc=0, scale=1)
    a_corr_ratio_calc = \
        1 - sci.stats.f.cdf(F_corr_ratio_calc, dfn, dfd, loc=0, scale=1)
    conclusion_corr_ratio_sign = 'significance' \
        if F_corr_ratio_calc >= F_corr_ratio_table else 'not significance'
    # доверительный интервал корреляционного отношения
    if F_corr_ratio_calc >= F_corr_ratio_table:
        f1 = round((K_X-1+n_X*corr_ratio**2)**2/(K_X-1+2*n_X*corr_ratio**2))
        f2 = n_X - K_X
        z1 = (n_X - K_X) / n_X * corr_ratio**2 / (1 - corr_ratio**2) * \
            1/sci.stats.f.ppf(p_level, f1, f2, loc=0, scale=1) - (K_X - 1)/n_X
        z2 = (n_X - K_X) / n_X * corr_ratio**2 / (1 - corr_ratio**2) * \
            1/sci.stats.f.ppf(1 - p_level, f1, f2, loc=0, scale=1) - \
                (K_X - 1)/n_X
        corr_ratio_conf_int_low = sqrt(z1) if sqrt(z1) >= 0 else 0
        corr_ratio_conf_int_high = sqrt(z2) if sqrt(z2) <= 1 else 1
    else:
        corr_ratio_conf_int_low = corr_ratio_conf_int_high = '-'    
    # оценка тесноты связи
    if scale=='Cheddok':
        conclusion_corr_ratio_scale = scale + ': ' + \
            Cheddock_scale_check(corr_ratio, name=chr(951))
    elif scale=='Evans':
        conclusion_corr_ratio_scale = scale + ': ' + \
            Evans_scale_check(corr_ratio, name=chr(951))
    # формируем результат            
    result = pd.DataFrame({
        'notation': (chr(951)),
        'coef_value': (corr_ratio),
        'coef_value_squared': (corr_ratio**2),
        'p_level': (p_level),
        'a_level': (a_level),
        'F_calc': (F_corr_ratio_calc),
        'F_table': (F_corr_ratio_table),
        'F_calc >= F_table': (F_corr_ratio_calc >= F_corr_ratio_table),
        'a_calc': (a_corr_ratio_calc),
        'a_calc <= a_level': (a_corr_ratio_calc <= a_level),
        'significance_check': (conclusion_corr_ratio_sign),
        'conf_int_low': (corr_ratio_conf_int_low),
        'conf_int_high': (corr_ratio_conf_int_high),
        'scale': (conclusion_corr_ratio_scale)
        },
        index=['Correlation ratio'])
    
    return result      

    

#------------------------------------------------------------------------------
#   Функция line_corr_sign_check
#
#   Функция для проверка значимости линейной корреляционной связи
#------------------------------------------------------------------------------

def line_corr_sign_check(
        X, Y, 
        p_level:        float = 0.95, 
        orientation:    str = 'XY'):
    
    """
    Расчет и проверка значимости линейной корреляционной связи.
        
    Args:
        X, Y:                      
            массивы исходных данных.
            
        p_level (float, optional):    
            доверительная вероятность. 
            Defaults to 0.95.

        orientation (str, optional):               
            направление связи ('XY' или 'YX').
            Defaults to 'XY'
            
    Returns:
        result (pd.core.frame.DataFrame):
            результат.
        
    """
    
    a_level = 1 - p_level
    X = np.array(X)
    Y = np.array(Y)
    n_X = len(X)
    n_Y = len(Y)    
    # коэффициент корреляции
    corr_coef = sci.stats.pearsonr(X, Y)[0]
    # корреляционное отношение
    try:
        if orientation!='XY' and orientation!='YX':
            raise ValueError("Error! Incorrect orientation!")
        if orientation=='XY':
            corr_ratio = corr_ratio_check(X, Y, orientation='XY', \
                scale='Evans')['coef_value'].values[0]
        elif orientation=='YX':
            corr_ratio = corr_ratio_check(X, Y, orientation='YX', \
                scale='Evans')['coef_value'].values[0]
    except ValueError as err:
        print(err)
    # число интервалов группировки
    group_int_number = lambda n: round (3.31*log(n_X, 10)+1) \
        if round (3.31*log(n_X, 10)+1) >=2 else 2
    K_X = group_int_number(n_X)
    # проверка гипотезы о значимости линейной корреляционной связи
    if corr_ratio >= abs(corr_coef):
        F_line_corr_sign_calc = \
            (n_X - K_X)/(K_X - 2) * \
                (corr_ratio**2 - corr_coef**2) / (1 - corr_ratio**2)
        dfn = K_X - 2
        dfd = n_X - K_X
        F_line_corr_sign_table = \
            sci.stats.f.ppf(p_level, dfn, dfd, loc=0, scale=1)
        comparison_F_calc_table = \
            F_line_corr_sign_calc >= F_line_corr_sign_table
        a_line_corr_sign_calc = \
            1-sci.stats.f.cdf(F_line_corr_sign_calc, dfn, dfd, loc=0, scale=1)
        comparison_a_calc_a_level = a_line_corr_sign_calc <= a_level
        conclusion_null_hypothesis_check = 'accepted' \
            if F_line_corr_sign_calc < F_line_corr_sign_table else 'unaccepted'
        conclusion_line_corr_sign = 'linear' \
            if conclusion_null_hypothesis_check == 'accepted' else 'non linear'
    else:
        F_line_corr_sign_calc = ''
        F_line_corr_sign_table = ''
        comparison_F_calc_table = ''
        a_line_corr_sign_calc = ''
        comparison_a_calc_a_level = ''
        conclusion_null_hypothesis_check = \
            'Attention! The correlation ratio is less than the correlation coefficient'
        conclusion_line_corr_sign = '-'
    
    # формируем результат            
    result = pd.DataFrame({
        'corr.coef.': (corr_coef),
        'corr.ratio.': (corr_ratio),
        'null hypothesis': ('r\u00b2 = ' + chr(951) + '\u00b2'),
        'p_level': (p_level),
        'a_level': (a_level),
        'F_calc': (F_line_corr_sign_calc),
        'F_table': (F_line_corr_sign_table),
        'F_calc >= F_table': (comparison_F_calc_table),
        'a_calc': (a_line_corr_sign_calc),
        'a_calc <= a_level': (comparison_a_calc_a_level),
        'null_hypothesis_check': (conclusion_null_hypothesis_check),
        'significance_line_corr_check': (conclusion_line_corr_sign),
        },
        index=['Significance of linear correlation'])
    
    return result



#------------------------------------------------------------------------------
#   Функция rank_corr_coef_check
#
#   Расчет и проверка значимости коэффициентов ранговой корреляции Кендалла и 
#   Спирмена
#------------------------------------------------------------------------------

def rank_corr_coef_check(
        X, Y, 
        p_level:    float = 0.95, 
        scale:      str = 'Evans'):

    """
    Расчет и проверка значимости коэффициентов ранговой корреляции Кендалл и 
    Спирмена.
        
    Args:
        X, Y:                      
            массивы исходных данных.
            
        p_level (float, optional):    
            доверительная вероятность. 
            Defaults to 0.95.

        scale (str, optional):               
            шкала для оценки тесноты связи.
            Defaults to 'Evans'.
            
    Returns:
        result (pd.core.frame.DataFrame):
            результат.
    
    """
    
    a_level = 1 - p_level
    X = np.array(X)
    Y = np.array(Y)
    n_X = len(X)
    n_Y = len(Y)
    # коэффициент ранговой корреляции Кендалл
    rank_corr_coef_tau, a_rank_corr_coef_tau_calc = sps.kendalltau(X, Y)
    # коэффициент ранговой корреляции Спирмена
    rank_corr_coef_spearman, a_rank_corr_coef_spearman_calc = \
        sps.spearmanr(X, Y)
    # критические значения коэффициентов
    if n_X >= 10:
        # табл.значение квантиля норм.распр.
        u_p_tau = sps.norm.ppf(p_level, 0, 1)    
        rank_corr_coef_tau_crit_value = \
            u_p_tau * sqrt(2*(2*n_X + 5) / (9*n_X*(n_X-1)))
        u_p_spearman = sps.norm.ppf((1+p_level)/2, 0, 1)
        rank_corr_coef_spearman_crit_value = u_p_spearman * 1/sqrt(n_X-1)
    else:
        rank_corr_coef_tau_crit_value = '-'
        rank_corr_coef_spearman_crit_value = '-'
    # проверка гипотезы о значимости
    conclusion_tau = 'significance' if a_rank_corr_coef_tau_calc <= a_level \
        else 'not significance'
    conclusion_spearman = 'significance' if \
        a_rank_corr_coef_spearman_calc <= a_level else 'not significance'
    # оценка тесноты связи
    if scale=='Cheddok':
        conclusion_scale_tau = scale + ': ' + \
            Cheddock_scale_check(rank_corr_coef_tau)
        conclusion_scale_spearman = scale + ': ' + \
            Cheddock_scale_check(rank_corr_coef_spearman)
    elif scale=='Evans':
        conclusion_scale_tau = scale + ': ' + \
            Evans_scale_check(rank_corr_coef_tau)
        conclusion_scale_spearman = scale + ': ' + \
            Evans_scale_check(rank_corr_coef_spearman)
    # доверительные интервалы (только для коэффициента Кендалла - см.[Айвазян, т.2, с.116])
    if conclusion_tau == 'significance':
        rank_corr_coef_tau_delta = sps.norm.ppf((1+p_level)/2, 0, 1) * \
            sqrt(2/n_X * (1 - rank_corr_coef_tau**2))
        rank_corr_coef_tau_int_low = rank_corr_coef_tau - \
            rank_corr_coef_tau_delta if rank_corr_coef_tau - rank_corr_coef_tau_delta else 0
        rank_corr_coef_tau_int_high = rank_corr_coef_tau + \
            rank_corr_coef_tau_delta if rank_corr_coef_tau + \
                rank_corr_coef_tau_delta <= 1 else 1
    # формируем результат            
    result = pd.DataFrame({
        'name': ('Kendall', 'Spearman'),
        'notation': (chr(964), chr(961)),
        'coef_value': (rank_corr_coef_tau, rank_corr_coef_spearman),
        'p_level': (p_level),
        'a_level': (a_level),
        'a_calc': (a_rank_corr_coef_tau_calc, a_rank_corr_coef_spearman_calc),
        'a_calc <= a_level': (a_rank_corr_coef_tau_calc <= a_level, \
                              a_rank_corr_coef_spearman_calc <= a_level),
        'crit_value': (rank_corr_coef_tau_crit_value, \
                       rank_corr_coef_spearman_crit_value),
        'crit_value >= coef_value': (
            rank_corr_coef_tau >= rank_corr_coef_tau_crit_value if \
                rank_corr_coef_tau_crit_value != '-' else '-',
            rank_corr_coef_spearman >= rank_corr_coef_spearman_crit_value if \
                rank_corr_coef_spearman_crit_value != '-' else '-'),
        'significance_check': (conclusion_tau, conclusion_spearman),
        'conf_int_low': (
            rank_corr_coef_tau_int_low if conclusion_tau == 'significance' \
                else '-', '-'),
        'conf_int_high': (
            rank_corr_coef_tau_int_high if \
                conclusion_tau == 'significance' else '-', '-'),
        'scale': (conclusion_scale_tau, conclusion_scale_spearman)
        })
    
    return result




#==============================================================================
#               ФУНКЦИИ ДЛЯ РЕГРЕССИОННОГО АНАЛИЗА
#==============================================================================


#------------------------------------------------------------------------------
#   Функция regression_error_metrics
#------------------------------------------------------------------------------

def regression_error_metrics(model=None, Yfact=None, Ycalc=None, model_name=''):
        
    if not(model==None):
        model_fit = model.fit()
        Ycalc = model_fit.predict()
        n_fit = model_fit.nobs
        Yfact = model.endog
        
        Y_mean = np.mean(Yfact)
        SSR = np.sum((Ycalc - Y_mean)**2)
        SSE = np.sum((Yfact - Ycalc)**2)
        SST = np.sum((Yfact - Y_mean)**2)
        
        MSE = (1/n_fit) * SSE
        RMSE = sqrt(MSE)
        MAE = (1/n_fit) * np.sum(abs(Yfact-Ycalc))
        MSPE = (1/n_fit) * np.sum(((Yfact-Ycalc)/Yfact)**2)
        MAPE = (1/n_fit) * np.sum(abs((Yfact-Ycalc)/Yfact))
        RMSLE = sqrt((1/n_fit) * np.sum((np.log(Yfact + 1)-np.log(Ycalc + 1))**2))
        R2 = 1 - SSE/SST
        
    
    else:
        Yfact = np.array(Yfact)
        Ycalc = np.array(Ycalc)
        n_fit = len(Yfact)
        
        Y_mean = np.mean(Yfact)
        SSR = np.sum((Ycalc - Y_mean)**2)
        SSE = np.sum((Yfact - Ycalc)**2)
        SST = np.sum((Yfact - Y_mean)**2)
        
        MSE = (1/n_fit) * SSE
        RMSE = sqrt(MSE)
        MAE = (1/n_fit) * np.sum(abs(Yfact-Ycalc))
        MSPE = (1/n_fit) *  np.sum(((Yfact-Ycalc)/Yfact)**2)
        MAPE = (1/n_fit) *  np.sum(abs((Yfact-Ycalc)/Yfact))        
        RMSLE = sqrt((1/n_fit) * np.sum((np.log(Yfact + 1)-np.log(Ycalc + 1))**2))
        R2 = 1 - SSE/SST
    
    model_error_metrics = {
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE,
        'MSPE': MSPE,
        'MAPE': MAPE,
        'RMSLE': RMSLE,
        'R2': R2}
    
    result = pd.DataFrame({
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE,
        'MSPE': "{:.3%}".format(MSPE),
        'MAPE': "{:.3%}".format(MAPE),
        'RMSLE': RMSLE,
        'R2': R2},
        index=[model_name])        
        
    return model_error_metrics, result



#------------------------------------------------------------------------------
#   Функция ANOVA_table_regression_model
#------------------------------------------------------------------------------

def ANOVA_table_regression_model(model):
    n = int(model.nobs)
    p = int(model.df_model)
    SSR = model.ess
    SSE = model.ssr
    SST = model.centered_tss

    result = pd.DataFrame({
        'sources_of_variation': ('regression (SSR)', 'deviation from regression (SSE)', 'total (SST)'),
        'sum_of_squares': (SSR, SSE, SST),
        'degrees_of_freedom': (p, n-p-1, n-1)})
    result['squared_error'] = result['sum_of_squares'] / result['degrees_of_freedom']
    R2 = 1 - result.loc[1, 'sum_of_squares'] / result.loc[2, 'sum_of_squares']
    F_calc_adequacy = result.loc[2, 'squared_error'] / result.loc[1, 'squared_error']
    F_calc_determ_check = result.loc[0, 'squared_error'] / result.loc[1, 'squared_error']
    result['F-ratio'] = (F_calc_determ_check, F_calc_adequacy, '')
    
    return result



#------------------------------------------------------------------------------
#   Функция regression_model_adequacy_check
#------------------------------------------------------------------------------


def regression_model_adequacy_check(
    model_fit,
    p_level: float=0.95,
    model_name=''):
    
    a_level = 1 - p_level
    
    n = int(model_fit.nobs)
    p = int(model_fit.df_model)    # Число степеней свободы регрессии, равно числу переменных модели (за исключением константы, если она присутствует)
    
    SST = model_fit.centered_tss    # SST (Sum of Squared Total)
    dfT = n-1
    MST = SST / dfT

    SSE = model_fit.ssr    # SSE (Sum of Squared Error)
    dfE = n - p - 1
    MSE = SSE / dfE
    
    F_calc = MST / MSE
    F_table = sci.stats.f.ppf(p_level, dfT, dfE, loc=0, scale=1)
    a_calc = 1 - sci.stats.f.cdf(F_calc, dfT, dfE, loc=0, scale=1)
    conclusion_model_adequacy_check = 'adequacy' if F_calc >= F_table else 'adequacy'
    
    # формируем результат            
    result = pd.DataFrame({
        'SST': (SST),
        'SSE': (SSE),
        'dfT': (dfT),
        'dfE': (dfE),
        'MST': (MST),
        'MSE': (MSE),
        'p_level': (p_level),
        'a_level': (a_level),
        'F_calc': (F_calc),
        'F_table': (F_table),
        'F_calc >= F_table': (F_calc >= F_table),
        'a_calc': (a_calc),
        'a_calc <= a_level': (a_calc <= a_level),
        'adequacy_check': (conclusion_model_adequacy_check),
        },
        index=[model_name]
        )
    
    return result



#------------------------------------------------------------------------------
#   Функция determination_coef_check
#------------------------------------------------------------------------------

def determination_coef_check(
    model_fit,
    p_level: float=0.95):
    
    a_level = 1 - p_level
    
    R2 = model_fit.rsquared
    R2_adj = model_fit.rsquared_adj
    n = model_fit.nobs    # объем выборки
    p = model_fit.df_model    # Model degrees of freedom. The number of regressors p. Does not include the constant if one is present.
    
    F_calc = R2 / (1 - R2) * (n-p-1)/p
    df1 = int(p)
    df2 = int(n-p-1)
    F_table = sci.stats.f.ppf(p_level, df1, df2, loc=0, scale=1)
    a_calc = 1 - sci.stats.f.cdf(F_calc, df1, df2, loc=0, scale=1)
    conclusion_determ_coef_sign = 'significance' if F_calc >= F_table else 'not significance'
        
    # формируем результат            
    result = pd.DataFrame({
        'notation': ('R2'),
        'coef_value (R)': (sqrt(R2)),
        'coef_value_squared (R2)': (R2),
        'p_level': (p_level),
        'a_level': (a_level),
        'F_calc': (F_calc),
        'df1': (df1),
        'df2': (df2),
        'F_table': (F_table),
        'F_calc >= F_table': (F_calc >= F_table),
        'a_calc': (a_calc),
        'a_calc <= a_level': (a_calc <= a_level),
        'significance_check': (conclusion_determ_coef_sign),
        'conf_int_low': (''),
        'conf_int_high': ('')
        },
        index=['Coef. of determination'])
    return result



#------------------------------------------------------------------------------
#   Функция regression_coef_check
#------------------------------------------------------------------------------

def regression_coef_check(
    model_fit,
    notation_coef: list='',
    p_level: float=0.95):
    
    a_level = 1 - p_level
    
    # параметры модели (коэффициенты регрессии)
    model_params = model_fit.params
    # стандартные ошибки коэффициентов регрессии
    model_bse = model_fit.bse
    # проверка гипотезы о значимости регрессии
    t_calc = abs(model_params) / model_bse
    n = model_fit.nobs    # объем выборки
    p = model_fit.df_model    # Model degrees of freedom. The number of regressors p. Does not include the constant if one is present.
    df = int(n - p - 1)
    t_table = sci.stats.t.ppf((1 + p_level)/2 , df)
    a_calc = 2*(1-sci.stats.t.cdf(t_calc, df, loc=0, scale=1))
    conclusion_ = ['significance' if elem else 'not significance' for elem in (t_calc >= t_table).values]
        
    # доверительный интервал коэффициента регрессии
    conf_int_low = model_params - t_table*model_bse
    conf_int_high = model_params + t_table*model_bse
    
    # формируем результат            
    result = pd.DataFrame({
        'notation': (notation_coef),
        'coef_value': (model_params),
        'std_err': (model_bse),
        'p_level': (p_level),
        'a_level': (a_level),
        't_calc': (t_calc),
        'df': (df),
        't_table': (t_table),
        't_calc >= t_table': (t_calc >= t_table),
        'a_calc': (a_calc),
        'a_calc <= a_level': (a_calc <= a_level),
        'significance_check': (conclusion_),
        'conf_int_low': (conf_int_low),
        'conf_int_high': (conf_int_high),
        })
    
    return result



#------------------------------------------------------------------------------
#   Функция Goldfeld_Quandt_test
#------------------------------------------------------------------------------

def Goldfeld_Quandt_test(
    model_fit,
    p_level: float=0.95,
    model_name=''):
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sms.het_goldfeldquandt(model_fit.resid, model_fit.model.exog)
    test_result = lzip(['F_statistic', 'p_calc'], test)    # распаковка результатов теста
    # расчетное значение статистики F-критерия
    F_calc_tuple = test_result[0]
    F_statistic = F_calc_tuple[1]
    # расчетный уровень значимости
    p_calc_tuple = test_result[1]
    p_calc = p_calc_tuple[1]
    # вывод
    conclusion_test = 'heteroscedasticity' if p_calc < a_level else 'not heteroscedasticity'
    
    result = pd.DataFrame({
        'test': ('Goldfeld–Quandt test'),
        'p_level': (p_level),
        'a_level': (a_level),
        'F_statistic': (F_statistic),
        'p_calc': (p_calc),
        'p_calc < a_level': (p_calc < a_level),
        'heteroscedasticity_check': (conclusion_test)
        },
        index=[model_name])
    
    return result



#------------------------------------------------------------------------------
#   Функция Breush_Pagan_test
#------------------------------------------------------------------------------

def Breush_Pagan_test(
    model_fit,
    p_level: float=0.95,
    model_name=''):
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sms.het_breuschpagan(model_fit.resid, model_fit.model.exog)
    name = ['Lagrange_multiplier_statistic', 'p_calc_LM', 'F_statistic', 'p_calc']
    test_result = lzip(name, test)    # распаковка результатов теста
    # расчетное значение статистики теста множителей Лагранжа
    LM_calc_tuple = test_result[0]
    Lagrange_multiplier_statistic = LM_calc_tuple[1]
    # расчетный уровень значимости статистики теста множителей Лагранжа
    p_calc_LM_tuple = test_result[1]
    p_calc_LM = p_calc_LM_tuple[1]
    # расчетное значение F-статистики гипотезы о том, что дисперсия ошибки не зависит от x
    F_calc_tuple = test_result[2]
    F_statistic = F_calc_tuple[1]
    # расчетный уровень значимости F-статистики
    p_calc_tuple = test_result[3]
    p_calc = p_calc_tuple[1]
    # вывод
    conclusion_test = 'heteroscedasticity' if p_calc < a_level else 'not heteroscedasticity'

    # вывод
    conclusion_test = 'heteroscedasticity' if p_calc < a_level else 'not heteroscedasticity'
    
    result = pd.DataFrame({
        'test': ('Breush-Pagan test'),
        'p_level': (p_level),
        'a_level': (a_level),
        'Lagrange_multiplier_statistic': (Lagrange_multiplier_statistic),
        'p_calc_LM': (p_calc_LM),
        'p_calc_LM < a_level': (p_calc_LM < a_level),
        'F_statistic': (F_statistic),
        'p_calc': (p_calc),
        'p_calc < a_level': (p_calc < a_level),
        'heteroscedasticity_check': (conclusion_test)
        },
        index=[model_name])
    
    return result



#------------------------------------------------------------------------------
#   Функция White_test
#------------------------------------------------------------------------------

def White_test(
    model_fit,
    p_level: float=0.95,
    model_name=''):
    
    a_level = 1 - p_level
    
    # реализация теста
    test = sms.het_white(model_fit.resid, model_fit.model.exog)
    name = ['Lagrange_multiplier_statistic', 'p_calc_LM', 'F_statistic', 'p_calc']
    test_result = lzip(name, test)    # распаковка результатов теста
    # расчетное значение статистики теста множителей Лагранжа
    LM_calc_tuple = test_result[0]
    Lagrange_multiplier_statistic = LM_calc_tuple[1]
    # расчетный уровень значимости статистики теста множителей Лагранжа
    p_calc_LM_tuple = test_result[1]
    p_calc_LM = p_calc_LM_tuple[1]
    # расчетное значение F-статистики гипотезы о том, что дисперсия ошибки не зависит от x
    F_calc_tuple = test_result[2]
    F_statistic = F_calc_tuple[1]
    # расчетный уровень значимости F-статистики
    p_calc_tuple = test_result[3]
    p_calc = p_calc_tuple[1]
    # вывод
    conclusion_test = 'heteroscedasticity' if p_calc < a_level else 'not heteroscedasticity'

    # вывод
    conclusion_test = 'heteroscedasticity' if p_calc < a_level else 'not heteroscedasticity'
    
    result = pd.DataFrame({
        'test': ('White test'),
        'p_level': (p_level),
        'a_level': (a_level),
        'Lagrange_multiplier_statistic': (Lagrange_multiplier_statistic),
        'p_calc_LM': (p_calc_LM),
        'p_calc_LM < a_level': (p_calc_LM < a_level),
        'F_statistic': (F_statistic),
        'p_calc': (p_calc),
        'p_calc < a_level': (p_calc < a_level),
        'heteroscedasticity_check': (conclusion_test)
        },
        index=[model_name])
    
    return result



#------------------------------------------------------------------------------
#   Функция Durbin_Watson_test
#------------------------------------------------------------------------------

def Durbin_Watson_test(
    data,
    m = None,
    p_level: float=0.95):
    
    a_level = 1 - p_level
    data = np.array(data)
    n = len(data)
    
    # расчетное значение статистики критерия
    DW_calc = sms.stattools.durbin_watson(data)
    
    # табличное значение статистики критерия
    if (n >= 15) and (n <= 100):
        # восстанавливаем структуру DataFrame из csv-файла
        DW_table_df = pd.read_csv(
            filepath_or_buffer='table/Durbin_Watson_test_table.csv',
            sep=';',
            #index_col='n'
            )
                            
        DW_table_df = DW_table_df.rename(columns={'Unnamed: 0': 'n'})
        DW_table_df = DW_table_df.drop([0, 1, 2])
        
        for col in DW_table_df.columns:
            DW_table_df[col] = pd.to_numeric(DW_table_df[col], errors='ignore')
            
        DW_table_df = DW_table_df.set_index('n')

        DW_table_df.columns = pd.MultiIndex.from_product(
            [['p=0.95', 'p=0.975', 'p=0.99'],
            ['m=1', 'm=2', 'm=3', 'm=4', 'm=5'],
            ['dL','dU']])    
        
        # интерполяция табличных значений
        key = [f'p={p_level}', f'm={m}']
        f_lin_L = sci.interpolate.interp1d(DW_table_df.index, DW_table_df[tuple(key + ['dL'])])
        f_lin_U = sci.interpolate.interp1d(DW_table_df.index, DW_table_df[tuple(key + ['dU'])])
        DW_table_L = float(f_lin_L(n))
        DW_table_U = float(f_lin_U(n))
                   
        # проверка гипотезы
        Durbin_Watson_scale = {
            1: DW_table_L,
            2: DW_table_U,
            3: 4 - DW_table_U,
            4: 4 - DW_table_L,
            5: 4}
        
        Durbin_Watson_comparison = {
            1: ['0 ≤ DW_calc < DW_table_L',                   'H1: r > 0'],
            2: ['DW_table_L ≤ DW_calc ≤ DW_table_U',          'uncertainty'],
            3: ['DW_table_U < DW_calc < 4 - DW_table_U',      'H0: r = 0'],
            4: ['4 - DW_table_U ≤ DW_calc ≤ 4 - DW_table_L',  'uncertainty'],
            5: ['4 - DW_table_L < DW_calc ≤ 4',               'H1: r < 0']}
        
        r_scale = list(Durbin_Watson_scale.values())
        for i, elem in enumerate(r_scale):
            if DW_calc <= elem:
                key_scale = list(Durbin_Watson_scale.keys())[i]
                comparison = Durbin_Watson_comparison[key_scale][0]
                conclusion = Durbin_Watson_comparison[key_scale][1]
                break
           
    elif n < 15:        
        comparison = '-'
        conclusion = 'count less than 15'
    else:
        comparison = '-'
        conclusion = 'count more than 100'
    
    
    # формируем результат            
    result = pd.DataFrame({
        'n': (n),
        'm': (m),
        'p_level': (p_level),
        'a_level': (a_level),
        'DW_calc': (DW_calc),
        'ρ': (1 - DW_calc/2),
        'DW_table_L': (DW_table_L if (n >= 15) and (n <= 100) else '-'),
        'DW_table_U': (DW_table_U if (n >= 15) and (n <= 100) else '-'),
        'comparison of calculated and critical values': (comparison),
        'conclusion': (conclusion)
        },
        index=['Durbin-Watson_test'])
    
    
    return result



#------------------------------------------------------------------------------
#   Функция regression_pair_predict
#------------------------------------------------------------------------------


def regression_pair_predict(
    x_in,
    model_fit,
    regression_model,
    p_level: float=0.95):
    
    a_level = 1 - p_level
    
    X = pd.DataFrame(model_fit.model.exog)[1].values    # найти лучшее решение
    Y = model_fit.model.endog
    
    # вспомогательные величины
    n = int(model_fit.nobs)
    p = int(model_fit.df_model)
    
    SSE = model_fit.ssr    # SSE (Sum of Squared Error)
    dfE = n - p - 1
    MSE = SSE / dfE    # остаточная дисперсия
    
    Xmean = np.mean(X)
    SST_X = np.sum([(X[i] - Xmean)**2 for i in range(0, n)])
    
    t_table = sci.stats.t.ppf((1 + p_level)/2 , dfE)
    S2_y_calc_mean = MSE * (1/n + (x_in - Xmean)**2 / SST_X)
    S2_y_calc_predict = MSE * (1 + 1/n + (x_in - Xmean)**2 / SST_X)
        
    # прогнозируемое значение переменной Y
    y_calc=regression_model(x_in)
    # доверительный интервал средних значений переменной Y
    y_calc_mean_ci_low = y_calc - t_table*sqrt(S2_y_calc_mean)
    y_calc_mean_ci_upp = y_calc + t_table*sqrt(S2_y_calc_mean)
    # доверительный интервал индивидуальных значений переменной Y
    y_calc_predict_ci_low = y_calc - t_table*sqrt(S2_y_calc_predict)
    y_calc_predict_ci_upp = y_calc + t_table*sqrt(S2_y_calc_predict)
    
    result = y_calc, y_calc_mean_ci_low, y_calc_mean_ci_upp, y_calc_predict_ci_low, y_calc_predict_ci_upp
    
    return result


#------------------------------------------------------------------------------
#   Функция graph_regression_pair_predict_plot_sns
#------------------------------------------------------------------------------

def graph_regression_pair_predict_plot_sns(
    model_fit,
    regression_model_in,
    Xmin=None, Xmax=None, Nx=10,
    Ymin_graph=None, Ymax_graph=None,
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    y_label=None,
    label_fontsize=14, tick_fontsize=12, 
    label_legend_regr_model='', label_legend_fontsize=12,
    s=50,
    graph_size=(297/INCH, 210/INCH),
    result_output=True,
    file_name=None):
    
    # фактические данные
    X = pd.DataFrame(model_fit.model.exog)[1].values    # найти лучшее решение
    Y = model_fit.model.endog
    X = np.array(X)
    Y = np.array(Y)
    
    # границы
    if not(Xmin) and not(Xmax):
        Xmin=min(X)
        Xmax=max(X)
        Xmin_graph=min(X)*0.99
        Xmax_graph=max(X)*1.01
    else:
        Xmin_graph=Xmin
        Xmax_graph=Xmax
    
    if not(Ymin_graph) and not(Ymax_graph):
        Ymin_graph=min(Y)*0.99
        Ymax_graph=max(Y)*1.01       
    
    # формируем DataFrame данных
    Xcalc = np.linspace(Xmin, Xmax, num=Nx)
    Ycalc = regression_model_in(Xcalc)
    
    result_df = pd.DataFrame(
        [regression_pair_predict(elem, model_fit, regression_model=regression_model_in) for elem in Xcalc],
        columns=['y_calc', 'y_calc_mean_ci_low', 'y_calc_mean_ci_upp', 'y_calc_predict_ci_low', 'y_calc_predict_ci_upp'])
    result_df.insert(0, 'x_calc', Xcalc)
            
    # заголовки графика
    fig, axes = plt.subplots(figsize=graph_size)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    # фактические данные
    sns.scatterplot(
        x=X, y=Y,
        label='data',
        s=s,
        color='red',
        ax=axes)
    
    # график регрессионной модели
    sns.lineplot(
        x=Xcalc, y=Ycalc,
        color='blue',
        linewidth=2,
        legend=True,
        label=label_legend_regr_model,
        ax=axes)
    
    # доверительный интервал средних значений переменной Y
    Mean_ci_low = result_df['y_calc_mean_ci_low']
    plt.plot(
        result_df['x_calc'], Mean_ci_low,
        color='magenta', linestyle='--', linewidth=1,
        label='confidence interval of mean values Y')
    
    Mean_ci_upp = result_df['y_calc_mean_ci_upp']
    plt.plot(
        result_df['x_calc'], Mean_ci_upp,
        color='magenta', linestyle='--', linewidth=1)
    
    # доверительный интервал индивидуальных значений переменной Y
    Predict_ci_low = result_df['y_calc_predict_ci_low']
    plt.plot(
        result_df['x_calc'], Predict_ci_low,
        color='orange', linestyle='-.', linewidth=2,
        label='confidence interval of individual values Y')
    
    Predict_ci_upp = result_df['y_calc_predict_ci_upp']
    plt.plot(
        result_df['x_calc'], Predict_ci_upp,
        color='orange', linestyle='-.', linewidth=2)
    
        
    axes.set_xlim(Xmin_graph, Xmax_graph)
    axes.set_ylim(Ymin_graph, Ymax_graph)        
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    #axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})
        
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    if result_output:
        return result_df
    else:
        return


#------------------------------------------------------------------------------
#   Функция simple_approximation
#------------------------------------------------------------------------------

def simple_approximation(
    X_in, Y_in,
    models_list_in,
    p0_dict_in=None,
    Xmin=None, Xmax=None,
    Ymin=None, Ymax=None,
    nx_in=100,
    DecPlace=4,
    result_table=False, value_table_calc=False, value_table_graph=False,
    title_figure=None, title_figure_fontsize=16,
    title_axes=None, title_axes_fontsize=18,
    x_label=None, y_label=None,
    linewidth=2,
    label_fontsize=14, tick_fontsize=12, label_legend_fontsize=12,
    color_list_in=None,
    b0_formatter=None, b1_formatter=None, b2_formatter=None, b3_formatter=None,
    graph_size=(420/INCH, 297/INCH),
    file_name=None):
    
    # Equations
    linear_func = lambda x, b0, b1: b0 + b1*x
    quadratic_func = lambda x, b0, b1, b2: b0 + b1*x + b2*x**2
    qubic_func = lambda x, b0, b1, b2, b3: b0 + b1*x + b2*x**2 + b3*x**3
    power_func = lambda x, b0, b1: b0 * x**b1
    exponential_func = lambda x, b0, b1: b0 * np.exp(b1*x)
    logarithmic_func = lambda x, b0, b1: b0 + b1*np.log(x)
    hyperbolic_func = lambda x, b0, b1: b0 + b1/x
    
    # Model reference
    p0_dict = {
        'linear':      [1, 1],
        'quadratic':   [1, 1, 1],
        'qubic':       [1, 1, 1, 1],
        'power':       [1, 1],
        'exponential': [1, 1],
        'logarithmic': [1, 1],
        'hyperbolic':  [1, 1]}
    
    models_dict = {
        'linear':      [linear_func,      p0_dict['linear']],
        'quadratic':   [quadratic_func,   p0_dict['quadratic']],
        'qubic':       [qubic_func,       p0_dict['qubic']],
        'power':       [power_func,       p0_dict['power']],
        'exponential': [exponential_func, p0_dict['exponential']],
        'logarithmic': [logarithmic_func, p0_dict['logarithmic']],
        'hyperbolic':  [hyperbolic_func,  p0_dict['hyperbolic']]}
    
    models_df = pd.DataFrame({
        'func': (
            linear_func,
            quadratic_func,
            qubic_func,
            power_func,
            exponential_func,
            logarithmic_func,
            hyperbolic_func),
        'p0': (
            p0_dict['linear'],
            p0_dict['quadratic'],
            p0_dict['qubic'],
            p0_dict['power'],
            p0_dict['exponential'],
            p0_dict['logarithmic'],
            p0_dict['hyperbolic'])},
        index=['linear', 'quadratic', 'qubic', 'power', 'exponential', 'logarithmic', 'hyperbolic'])
        
    models_dict_keys_list = list(models_dict.keys())
    models_dict_values_list = list(models_dict.values())
        
    # Initial guess for the parameters
    if p0_dict_in:
        p0_dict_in_keys_list = list(p0_dict_in.keys())
        for elem in models_dict_keys_list:
            if elem in p0_dict_in_keys_list:
                models_dict[elem][1] = p0_dict_in[elem]
            
    # Calculations
    X_fact = np.array(X_in)
    Y_fact = np.array(Y_in)
    
    nx = 100 if not(nx_in) else nx_in
    hx = (Xmax - Xmin)/(nx - 1)
    X_calc_graph = np.linspace(Xmin, Xmax, nx)
    
    parameters_list = list()
    models_list = list()
    
    error_metrics_df = pd.DataFrame(columns=['MSE', 'RMSE', 'MAE', 'MSPE', 'MAPE', 'RMSLE', 'R2'])
    Y_calc_graph_df = pd.DataFrame({'X': X_calc_graph})
    Y_calc_df = pd.DataFrame({
        'X_fact': X_fact,
        'Y_fact': Y_fact})
    
    for elem in models_list_in:
        if elem in models_dict_keys_list:
            func = models_dict[elem][0]
            p0 = models_dict[elem][1]
            popt_, _ = curve_fit(func, X_fact, Y_fact, p0=p0)
            models_dict[elem].append(popt_)
            Y_calc_graph = func(X_calc_graph, *popt_)
            Y_calc = func(X_fact, *popt_)
            Y_calc_graph_df[elem] = Y_calc_graph
            Y_calc_df[elem] = Y_calc
            parameters_list.append(popt_)
            models_list.append(elem)
            (model_error_metrics, result_error_metrics) = regression_error_metrics(Yfact=Y_fact, Ycalc=Y_calc_df[elem], model_name=elem)
            error_metrics_df = pd.concat([error_metrics_df, result_error_metrics])
                
    parameters_df = pd.DataFrame(parameters_list,
                                 index=models_list)
    parameters_df = parameters_df.add_prefix('b')
    result_df = parameters_df.join(error_metrics_df)
                        
    # Legend for a linear model
    if "linear" in models_list_in:
        b0_linear = round(result_df.loc["linear", "b0"], DecPlace)
        b0_linear_str = str(b0_linear)
        b1_linear = round(result_df.loc["linear", "b1"], DecPlace)
        b1_linear_str = f' + {b1_linear}' if b1_linear > 0 else f' - {abs(b1_linear)}'
        R2_linear = round(result_df.loc["linear", "R2"], DecPlace)
        MSPE_linear = result_df.loc["linear", "MSPE"]
        MAPE_linear = result_df.loc["linear", "MAPE"]
        label_linear = 'linear: ' + r'$Y_{calc} = $' + b0_linear_str + b1_linear_str + f'{chr(183)}X' + ' ' + \
            r'$(R^2 = $' + f'{R2_linear}' + ', ' + f'MSPE = {MSPE_linear}' + ', ' + f'MAPE = {MAPE_linear})'
    
    # Legend for a quadratic model
    if "quadratic" in models_list_in:
        b0_quadratic = round(result_df.loc["quadratic", "b0"], DecPlace)
        b0_quadratic_str = str(b0_quadratic)
        b1_quadratic = result_df.loc["quadratic", "b1"]
        b1_quadratic_str = f' + {b1_quadratic:.{DecPlace}e}' if b1_quadratic > 0 else f' - {abs(b1_quadratic):.{DecPlace}e}'
        b2_quadratic = result_df.loc["quadratic", "b2"]
        b2_quadratic_str = f' + {b2_quadratic:.{DecPlace}e}' if b2_quadratic > 0 else f' - {abs(b2_quadratic):.{DecPlace}e}'
        R2_quadratic = round(result_df.loc["quadratic", "R2"], DecPlace)
        MSPE_quadratic = result_df.loc["quadratic", "MSPE"]
        MAPE_quadratic = result_df.loc["quadratic", "MAPE"]
        label_quadratic = 'quadratic: ' + r'$Y_{calc} = $' + b0_quadratic_str + b1_quadratic_str + f'{chr(183)}X' + b2_quadratic_str + f'{chr(183)}' + r'$X^2$' + ' ' + \
            r'$(R^2 = $' + f'{R2_quadratic}' + ', ' + f'MSPE = {MSPE_quadratic}' + ', ' + f'MAPE = {MAPE_quadratic})'
    
    # Legend for a qubic model
    if "qubic" in models_list_in:
        b0_qubic = round(result_df.loc["qubic", "b0"], DecPlace)
        b0_qubic_str = str(b0_qubic)
        b1_qubic = result_df.loc["qubic", "b1"]
        b1_qubic_str = f' + {b1_qubic:.{DecPlace}e}' if b1_qubic > 0 else f' - {abs(b1_qubic):.{DecPlace}e}'
        b2_qubic = result_df.loc["qubic", "b2"]
        b2_qubic_str = f' + {b2_qubic:.{DecPlace}e}' if b2_qubic > 0 else f' - {abs(b2_qubic):.{DecPlace}e}'
        b3_qubic = result_df.loc["qubic", "b3"]
        b3_qubic_str = f' + {b3_qubic:.{DecPlace}e}' if b3_qubic > 0 else f' - {abs(b3_qubic):.{DecPlace}e}'
        R2_qubic = round(result_df.loc["qubic", "R2"], DecPlace)
        MSPE_qubic = result_df.loc["qubic", "MSPE"]
        MAPE_qubic = result_df.loc["qubic", "MAPE"]
        label_qubic = 'qubic: ' + r'$Y_{calc} = $' + b0_qubic_str + b1_qubic_str + f'{chr(183)}X' + \
            b2_qubic_str + f'{chr(183)}' + r'$X^2$' + b3_qubic_str + f'{chr(183)}' + r'$X^3$' + ' ' + \
            r'$(R^2 = $' + f'{R2_qubic}' + ', ' + f'MSPE = {MSPE_qubic}' + ', ' + f'MAPE = {MAPE_qubic})'
    
    # Legend for a power model
    if "power" in models_list_in:
        b0_power = round(result_df.loc["power", "b0"], DecPlace)
        b0_power_str = str(b0_power)
        b1_power = round(result_df.loc["power", "b1"], DecPlace)
        b1_power_str = str(b1_power)
        R2_power = round(result_df.loc["power", "R2"], DecPlace)
        MSPE_power = result_df.loc["power", "MSPE"]
        MAPE_power = result_df.loc["power", "MAPE"]
        label_power = 'power: ' + r'$Y_{calc} = $' + b0_power_str + f'{chr(183)}' + r'$X$'
        for elem in b1_power_str:
            label_power = label_power + r'$^{}$'.format(elem)
        label_power = label_power  + ' ' + r'$(R^2 = $' + f'{R2_power}' + ', ' + f'MSPE = {MSPE_power}' + ', ' + f'MAPE = {MAPE_power})'
    
    # Legend for a exponential model
    if "exponential" in models_list_in:
        b0_exponential = round(result_df.loc["exponential", "b0"], DecPlace)
        b0_exponential_str = str(b0_exponential)
        b1_exponential = result_df.loc["exponential", "b1"]
        b1_exponential_str = f'{b1_exponential:.{DecPlace}e}'
        R2_exponential = round(result_df.loc["exponential", "R2"], DecPlace)
        MSPE_exponential = result_df.loc["exponential", "MSPE"]
        MAPE_exponential = result_df.loc["exponential", "MAPE"]
        label_exponential = 'exponential: ' + r'$Y_{calc} = $' + b0_exponential_str + f'{chr(183)}' + r'$e$'
        for elem in b1_exponential_str:
            label_exponential = label_exponential + r'$^{}$'.format(elem)
        label_exponential = label_exponential + r'$^{}$'.format(chr(183)) + r'$^X$' + ' ' + \
            r'$(R^2 = $' + f'{R2_exponential}' + ', ' + f'MSPE = {MSPE_exponential}' + ', ' + f'MAPE = {MAPE_exponential})'
    
    # Legend for a logarithmic model
    if "logarithmic" in models_list_in:
        b0_logarithmic = round(result_df.loc["logarithmic", "b0"], DecPlace)
        b0_logarithmic_str = str(b0_logarithmic)
        b1_logarithmic = round(result_df.loc["logarithmic", "b1"], DecPlace)
        b1_logarithmic_str = f' + {b1_logarithmic}' if b1_logarithmic > 0 else f' - {abs(b1_logarithmic)}'
        R2_logarithmic = round(result_df.loc["logarithmic", "R2"], DecPlace)
        MSPE_logarithmic = result_df.loc["logarithmic", "MSPE"]
        MAPE_logarithmic = result_df.loc["logarithmic", "MAPE"]
        label_logarithmic = 'logarithmic: ' + r'$Y_{calc} = $' + b0_logarithmic_str + b1_logarithmic_str + f'{chr(183)}ln(X)' + ' ' + \
            r'$(R^2 = $' + f'{R2_logarithmic}' + ', ' + f'MSPE = {MSPE_logarithmic}' + ', ' + f'MAPE = {MAPE_logarithmic})'
    
    # Legend for a hyperbolic model
    if "hyperbolic" in models_list_in:
        b0_hyperbolic = round(result_df.loc["hyperbolic", "b0"], DecPlace)
        b0_hyperbolic_str = str(b0_hyperbolic)
        b1_hyperbolic = round(result_df.loc["hyperbolic", "b1"], DecPlace)
        b1_hyperbolic_str = f' + {b1_hyperbolic}' if b1_hyperbolic > 0 else f' - {abs(b1_hyperbolic)}'
        R2_hyperbolic = round(result_df.loc["hyperbolic", "R2"], DecPlace)
        MSPE_hyperbolic = result_df.loc["hyperbolic", "MSPE"]
        MAPE_hyperbolic = result_df.loc["hyperbolic", "MAPE"]
        label_hyperbolic = 'hyperbolic: ' + r'$Y_{calc} = $' + b0_hyperbolic_str + b1_hyperbolic_str + f' / X' + ' ' + \
            r'$(R^2 = $' + f'{R2_hyperbolic}' + ', ' + f'MSPE = {MSPE_hyperbolic}' + ', ' + f'MAPE = {MAPE_hyperbolic})'
        
    # Legends
    label_legend_dict = {
        'linear':      label_linear if "linear" in models_list_in else '',
        'quadratic':   label_quadratic if "quadratic" in models_list_in else '',
        'qubic':       label_qubic if "qubic" in models_list_in else '',
        'power':       label_power if "power" in models_list_in else '',
        'exponential': label_exponential if "exponential" in models_list_in else '',
        'logarithmic': label_logarithmic if "logarithmic" in models_list_in else '',
        'hyperbolic':  label_hyperbolic if "hyperbolic" in models_list_in else ''}
    
    # Graphics
    color_dict = {
        'linear':      'blue',
        'quadratic':   'green',
        'qubic':       'brown',
        'power':       'magenta',
        'exponential': 'cyan',
        'logarithmic': 'orange',
        'hyperbolic':  'grey'}
    
    if not(Xmin) and not(Xmax):
        Xmin = min(X_fact)*0.99
        Xmax = max(X_fact)*1.01
    if not(Ymin) and not(Ymax):
        Ymin = min(Y_fact)*0.99
        Ymax = max(Y_fact)*1.01     
        
    fig, axes = plt.subplots(figsize=(graph_size))
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    sns.scatterplot(
        x=X_fact, y=Y_fact,
        label='data',
        s=50,
        color='red',
        ax=axes)
       
    for elem in models_list_in:
        if elem in models_dict_keys_list:
            sns.lineplot(
                x=X_calc_graph, y=Y_calc_graph_df[elem],
                color=color_dict[elem],
                linewidth=linewidth,
                legend=True,
                label=label_legend_dict[elem],
                ax=axes)

    axes.set_xlim(Xmin, Xmax)
    axes.set_ylim(Ymin, Ymax)
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})

    plt.show()
    
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
    
    # result output
    output_list = [result_df, Y_calc_df, Y_calc_graph_df]
    result_list = [result_table, value_table_calc, value_table_graph]
    result = list()
    for i, elem in enumerate(result_list):
        if elem:
            result.append(output_list[i])
    
    # result display
    for elem in ['MSPE', 'MAPE']:
        result_df[elem] = result_df[elem].apply(lambda s: float(s[:-1]))
    b_formatter = [b0_formatter, b1_formatter, b2_formatter, b3_formatter]
    if result_table:
        display(result_df
                .style
                    .format(
                        precision=DecPlace, na_rep='-',
                        formatter={
                            'b0': b0_formatter if b0_formatter else '{:.4f}',
                            'b1': b1_formatter if b1_formatter else '{:.4f}',
                            'b2': b2_formatter if b2_formatter else '{:.4e}',
                            'b3': b3_formatter if b3_formatter else '{:.4e}'})
                    .highlight_min(color='green', subset=['MSE', 'RMSE', 'MAE', 'MSPE', 'MAPE'])
                    .highlight_max(color='red', subset=['MSE', 'RMSE', 'MAE', 'MSPE', 'MAPE'])
                    .highlight_max(color='green', subset='R2')
                    .highlight_min(color='red', subset='R2')
                    .applymap(lambda x: 'color: orange;' if abs(x) <= 10**(-(DecPlace-1)) else None, subset=parameters_df.columns))
    
    if value_table_calc:
        display(Y_calc_df)
    if value_table_graph:
        display(Y_calc_graph_df)
                    
    return result    



#==============================================================================
#               ФУНКЦИИ ДЛЯ АНАЛИЗА ВРЕМЕННЫХ РЯДОВ
#==============================================================================


#------------------------------------------------------------------------------
#   Функция adf_test
#------------------------------------------------------------------------------

def adf_test(timeseries, p_level=0.95):
    from statsmodels.tsa.stattools import adfuller
    a_level = 1 - p_level
    print("Dickey-Fuller Test:\n")
    
    (H0, H1) = ('H0: The time series is non-stationary',
                'H1: The time series is stationary')
    print(f'{H0}\n{H1}\n')
    
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used",],    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    
    a_calc = dfoutput['p-value']
    conclusion = \
        f'The time series is non-stationary (p_calc = {a_calc} >= a_level = {a_level})' if a_calc >= a_level else \
        f'The time series is stationary (a_calc = {a_calc} < a_level = {a_level})'
    print(conclusion, '\n')
    

#------------------------------------------------------------------------------
#   Функция kpss_test
#------------------------------------------------------------------------------    

def kpss_test(timeseries, regression='c', p_level=0.95):
    from statsmodels.tsa.stattools import kpss
    a_level = 1 - p_level
    print("KPSS Test:")
    
    H0 = \
        'H0: The data is stationary around a constant' if regression=='c' else \
        'H0: The data is stationary around a trend'
    H1 = 'H1: The data is non-stationary'
    print(f'{H0}\n{H1}\n')
    
    kpsstest = kpss(timeseries, regression=regression, nlags="auto", store=False)
    kpss_output = pd.Series(
        kpsstest[0:3], 
        index=["Test Statistic", "p-value", "Lags Used"])
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)
    
    a_calc = kpss_output['p-value']
    
    if regression=='c':
        conclusion = \
            f'The data is stationary around a constant (p_calc = {a_calc} >= a_level = {a_level})' if a_calc >= a_level else \
            f'The data is non-stationary (a_calc = {a_calc} < a_level = {a_level})'
    else:
        conclusion = \
            f'The data is stationary around a trend (p_calc = {a_calc} >= a_level = {a_level})' if a_calc >= a_level else \
            f'The data is non-stationary (a_calc = {a_calc} < a_level = {a_level})'
    print(conclusion, '\n')


#------------------------------------------------------------------------------
#   Функция adf_test_sm
#------------------------------------------------------------------------------

def ADF_test_sm(
    timeseries,
    regression: list=['n', 'c', 'ct', 'ctt'],    # “c” : constant only (default). “ct” : constant and trend. “ctt” : constant, and linear and quadratic trend. “n” : no constant, no trend.
    autolag: str='AIC',     # {“AIC”, “BIC”, “t-stat”, None}
    p_level: float=0.95):
    
    from statsmodels.tsa.stattools import adfuller
    
    a_level = 1 - p_level
    
    # https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    
    # Dickey-Fuller (ADF) test
    # https://ru.wikipedia.org/wiki/Тест_Дики_—_Фуллера, https://ru.wikipedia.org/wiki/%D0%95%D0%B4%D0%B8%D0%BD%D0%B8%D1%87%D0%BD%D1%8B%D0%B9_%D0%BA%D0%BE%D1%80%D0%B5%D0%BD%D1%8C
    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html#statsmodels.tsa.stattools.adfuller
    
    (H0, H1) = ('The time series is non-stationary',
                'The time series is stationary')
    display(pd.DataFrame({'ADF test': (H0, H1)},
                         index=('H0', 'H1')))
    
    result_dict = dict()    
    for elem in regression:
        result_temp = adfuller(
            timeseries,
            regression=elem,
            autolag=autolag)
        #print(result_temp)
        result_dict[elem] = [result_temp]
        
    a_calc_list = [result_dict[elem][0][1] for elem in regression]
        
    result = pd.DataFrame({
        'test': ('ADF test' for elem in regression),
        'p_level': (p_level),
        'a_level': (a_level),
        'regression': (regression),
        'test_statistic': (result_dict[elem][0][0] for elem in regression),
        'crit_value (10%)': (result_dict[elem][0][4]['10%'] for elem in regression),
        'crit_value (5%)': (result_dict[elem][0][4]['5%'] for elem in regression),
        'crit_value (1%)': (result_dict[elem][0][4]['1%'] for elem in regression),
        'a_calc': (a_calc_list),
        'a_calc >= a_level': (a_calc >= a_level for a_calc in a_calc_list),
        'null_hypothesis_check': ('accepted' if a_calc >= a_level else 'unaccepted' for a_calc in a_calc_list),
        'conclusion': ('The time series is non-stationary' if a_calc >= a_level else 'The time series is stationary' for a_calc in a_calc_list)})
        
    return result
    

#------------------------------------------------------------------------------
#   Функция kpss_test_sm
#------------------------------------------------------------------------------        

def KPSS_test_sm(
    timeseries,
    regression: list=['c', 'ct'],
    nlags='auto',
    p_level: float=0.95):
    
    from statsmodels.tsa.stattools import kpss
    
    a_level = 1 - p_level
    
    # https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    
    # Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test
    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.kpss.html
    
    (H0_1, H0_2, H1) = ('The data is stationary around a constant (if regression = "c")',
                        'The data is stationary around a trend (if regression = "ct")',
                        'The data is non-stationary')
    display(pd.DataFrame({'KPSS test': (H0_1, H0_2, H1)},
                         index=(r'H0_1', r'H0_2', 'H1')))
    
    import warnings
    warnings.filterwarnings('ignore')

    result_dict = dict()    
    for elem in regression:
        result_temp = kpss(
            timeseries,
            regression=elem,
            nlags=nlags)
        #print(result_temp)
        result_dict[elem] = [result_temp]
        
    a_calc_list = [result_dict[elem][0][1] for elem in regression]
        
    conclusion_list = list()
    for index, elem in enumerate(regression):
        if elem == 'c':
            conclusion_list.append('The data is stationary around a constant' if a_calc_list[index] >= a_level else 'The data is non-stationary')
        if elem == 'ct':
            conclusion_list.append('The data is stationary around a trend' if a_calc_list[index] >= a_level else 'The data is non-stationary')

    result = pd.DataFrame({
        'test': ('KPSS test' for elem in regression),
        'p_level': (p_level),
        'a_level': (a_level),
        'regression': (regression),
        'test_statistic': (result_dict[elem][0][0] for elem in regression),
        'crit_value (10%)': (result_dict[elem][0][3]['10%'] for elem in regression),
        'crit_value (5%)': (result_dict[elem][0][3]['5%'] for elem in regression),
        'crit_value (2.5%)': (result_dict[elem][0][3]['2.5%'] for elem in regression),
        'crit_value (1%)': (result_dict[elem][0][3]['1%'] for elem in regression),
        'a_calc': (a_calc_list),
        'a_calc >= a_level': (a_calc >= a_level for a_calc in a_calc_list),
        'null_hypothesis_check': ('accepted' if a_calc >= a_level else 'unaccepted' for a_calc in a_calc_list),
        'conclusion': (conclusion_list)})
            
    warnings.filterwarnings('default')        
    return result



#------------------------------------------------------------------------------
#   Функция ADF_KPSS_PP_test_pmdarima
#------------------------------------------------------------------------------

def ADF_KPSS_PP_test_pmdarima(
    timeseries,
    p_level: float=0.95):  
    
    a_level = 1 - p_level
    
    # Dickey–Fuller test (ADF)
    # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ADFTest.html#pmdarima.arima.ADFTest
    from pmdarima.arima import ADFTest  
    result_ADF = ADFTest(alpha = a_level)
    (a_calc_ADF, sig_ADF) = result_ADF.should_diff(timeseries)
    
    # Kwiatkowski–Phillips–Schmidt–Shin test (KPSS)
    # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.KPSSTest.html#pmdarima.arima.KPSSTest
    from pmdarima.arima import KPSSTest  
    result_KPSS = KPSSTest(alpha = a_level)
    (a_calc_KPSS, sig_KPSS) = result_KPSS.should_diff(timeseries)
    
    # Phillips–Perron test (PP)
    # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html#pmdarima.arima.PPTest
    from pmdarima.arima import PPTest  
    result_PP = PPTest(alpha = a_level)
    (a_calc_PP, sig_PP) = result_PP.should_diff(timeseries)
    
    # d-value
    # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ndiffs.html
    from pmdarima.arima.utils import ndiffs
    d_value_ADF = ndiffs(timeseries, test='adf')
    d_value_KPSS = ndiffs(timeseries, test='kpss')
    d_value_PP = ndiffs(timeseries, test='pp') 
    
    conclusion_list = ['The time series is non-stationary' if sig 
                       else 'The time series is stationary' 
                       for sig in [sig_ADF, sig_KPSS, sig_PP]]   
    
    result = pd.DataFrame({
        'test': ('ADF test', 'KPSS test', 'PP test'),
        'p_level': (p_level),
        'a_level': (a_level),
        'a_calc': (a_calc_ADF, a_calc_KPSS, a_calc_PP),
        'significance of a_calc': (sig_ADF, sig_KPSS, sig_PP),
        'conclusion': (conclusion_list),
        'd-value': (d_value_ADF, d_value_KPSS, d_value_PP)})
    
    return result


#------------------------------------------------------------------------------
#   Функция ADF_KPSS_PP_test_pmdarima
#------------------------------------------------------------------------------

def CH_OCSB_test_pmdarima(
    timeseries,
    m: int,       # Number of seasonal periods
    max_D: int    # Maximum number of seasonal differences allowed. The estimated value of D will not exceed max_D.
    ):
    
    # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.nsdiffs.html#pmdarima.arima.nsdiffs

    from pmdarima.arima import nsdiffs
    
    # Canova-Hansen test (CH)
    # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.CHTest.html#pmdarima.arima.CHTest
    D_CH = nsdiffs(timeseries, m=m, max_D=max_D, test='ch')
    
    # Osborn, Chui, Smith, and Birchenhall test (OCSB)
    # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.OCSBTest.html#pmdarima.arima.OCSBTest
    D_OCSB = nsdiffs(timeseries, m=m, max_D=max_D, test='ocsb')
    
    result = pd.DataFrame({
        'test': ('CH test', 'OCSB test'),
        'm': (m),
        'max_D': (max_D),
        'D-value': (D_CH, D_OCSB)})
    
    return result



#==============================================================================
#               SCIKIT-LEARN
#==============================================================================


#------------------------------------------------------------------------------
#   Функция для оценки моделей средствами **scikit-learn** 
#   (https://scikit-learn.ru/3-3-metrics-and-scoring-quantifying-the-quality-of-predictions/):
#------------------------------------------------------------------------------

def sklearn_error_metrics_regression(Y_test, Y_pred, model_name=''):
    import sklearn
        
    # проверка наличия отрицательных значений
    check_negative = True if (np.any(Y_test<0) or np.any(Y_pred<0)) else False
    
    # explained_variance_score (Explained variation / Объясненная вариации)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score
    # https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score
    # https://en.wikipedia.org/wiki/Explained_variation
    explained_variance_score = sklearn.metrics.explained_variance_score(Y_test, Y_pred)
    
    # max_error (Максимальная линейная ошибка)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error
    # https://scikit-learn.org/stable/modules/model_evaluation.html#max-error
    max_error = sklearn.metrics.max_error(Y_test, Y_pred)
    
    # neg_mean_absolute_error (Mean absolute error / Средняя абсолютная ошибка)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error
    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error
    # https://en.wikipedia.org/wiki/Mean_absolute_error
    MAE = sklearn.metrics.mean_absolute_error(Y_test, Y_pred)
    
    # neg_mean_squared_error (Mean squared error / Среднеквадратическая ошибка)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error
    # https://en.wikipedia.org/wiki/Mean_squared_error
    MSE = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
    
    # neg_root_mean_squared_error
    RMSE = np.sqrt(MSE)    
    
    # neg_mean_squared_log_error (Mean squared logarithmic error / Среднеквадратическая логарифмическая ошибка)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error
    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error
    
    MSLE = sklearn.metrics.mean_squared_log_error(Y_test, Y_pred) if not(check_negative) else '-'
    
    # neg_median_absolute_error (Median absolute error / Средняя абсолютная ошибка)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error
    # https://scikit-learn.org/stable/modules/model_evaluation.html#median-absolute-error
    median_absolute_error = sklearn.metrics.median_absolute_error(Y_test, Y_pred)
    
    # r2 (Coefficient of determination)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    # https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
    # https://en.wikipedia.org/wiki/Coefficient_of_determination
    R2 = sklearn.metrics.r2_score(Y_test, Y_pred)
    
    # neg_mean_poisson_deviance (Mean Poisson deviance / Среднее отклонение Пуассона)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html#sklearn.metrics.mean_poisson_deviance
    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance
    # https://en.wikipedia.org/wiki/Tweedie_distribution#The_Tweedie_deviance
    check_negative_mean_poisson_deviance = True if (np.any(Y_test<0) or np.any(Y_pred<=0)) else False
    mean_poisson_deviance = sklearn.metrics.mean_poisson_deviance(Y_test, Y_pred) if not(check_negative_mean_poisson_deviance) else '-'
    
    # neg_mean_gamma_deviance
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html#sklearn.metrics.mean_gamma_deviance
    check_negative_mean_gamma_deviance = True if (np.any(Y_test<0) or np.any(Y_pred<=0)) else False
    mean_gamma_deviance = sklearn.metrics.mean_gamma_deviance(Y_test, Y_pred) if not(check_negative_mean_gamma_deviance) else '-'
    
    # neg_mean_tweedie_deviance
    mean_tweedie_deviance = sklearn.metrics.mean_tweedie_deviance(Y_test, Y_pred) if not(check_negative) else '-'
    
    # neg_mean_absolute_percentage_error (Mean absolute percentage error / Средняя абсолютная ошибка в процентах)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html#sklearn.metrics.mean_absolute_percentage_error
    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error
    MAPE = sklearn.metrics.mean_absolute_percentage_error(Y_test, Y_pred)
    
    model_error_metrics = {
        'explained_variance_score': explained_variance_score,
        'max_error': max_error,
        'neg_mean_absolute_error': MAE,
        'neg_mean_squared_error': MSE,
        'neg_root_mean_squared_error': RMSE,
        'neg_mean_squared_log_error': MSLE,
        'median_absolute_error': median_absolute_error,
        'r2': R2,
        'neg_mean_poisson_deviance': mean_poisson_deviance,
        'neg_mean_gamma_deviance': mean_gamma_deviance,
        'neg_mean_tweedie_deviance': mean_tweedie_deviance,
        'neg_mean_absolute_percentage_error': MAPE}
    
    result = pd.DataFrame({
        'Expl.var.score': explained_variance_score,
        'Max error': max_error,
        'MAE': MAE,
        'MSE': MSE,
        'RMSE': RMSE,
        'MSLE': MSLE,
        'Med.abs.error': median_absolute_error,
        'R2': R2,
        'Mean Poisson dev.': mean_poisson_deviance,
        'Mean gamma dev.': mean_gamma_deviance,
        'Mean Tweedie dev.': mean_tweedie_deviance,
        'MAPE': MAPE},
        index=[model_name])        
        
    return model_error_metrics, result  



#==============================================================================
#               СТАРЫЕ ФУНКЦИИ
#==============================================================================


#------------------------------------------------------------------------------
#   Функция graph_scatterplot_sns
#------------------------------------------------------------------------------

def graph_scatterplot_sns(
    X, Y,
    Xmin=None, Xmax=None,
    Ymin=None, Ymax=None,
    color='orange',
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    y_label=None,
    label_fontsize=14, tick_fontsize=12,
    label_legend='', label_legend_fontsize=12,
    s=50,
    graph_size=(297/INCH, 210/INCH),
    file_name=None):
    
    X = np.array(X)
    Y = np.array(Y)
    
    if not(Xmin) and not(Xmax):
        Xmin=min(X)*0.99
        Xmax=max(X)*1.01
    if not(Ymin) and not(Ymax):
        Ymin=min(Y)*0.99
        Ymax=max(Y)*1.01       
    
    fig, axes = plt.subplots(figsize=graph_size)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    #data_df = data_XY_df
    sns.scatterplot(
        x=X, y=Y,
        label=label_legend,
        s=s,
        palette=['orange'], color=color,
        ax=axes)
    axes.set_xlim(Xmin, Xmax)
    axes.set_ylim(Ymin, Ymax)       
    axes.axvline(x = 0, color = 'k', linewidth = 1)
    axes.axhline(y = 0, color = 'k', linewidth = 1)
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    #axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})
        
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return    



#------------------------------------------------------------------------------
#   Функция graph_scatterplot_mpl_np
#------------------------------------------------------------------------------

def graph_plot_mpl_np(
    X, Y,
    Xmin_in=None, Xmax_in=None,
    Ymin_in=None, Ymax_in=None,
    color='orange',
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    y_label=None,
    label_fontsize=14, tick_fontsize=12,
    label_legend='', label_legend_fontsize=12,
    graph_size=(297/INCH, 210/INCH),
    file_name=None):
    
    sns.set_style("whitegrid")    # настройка цветовой гаммы
    fig, axes = plt.subplots(figsize=graph_size)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    if (Xmin_in != None) and (Xmax_in != None):
        Xmin=Xmin_in
        Xmax=Xmax_in
        axes.set_xlim(Xmin, Xmax)
            
    if (Ymin_in != None) and (Ymax_in != None):
        Ymin=Ymin_in
        Ymax=Ymax_in
        axes.set_ylim(Ymin, Ymax)        
    
    axes.plot(
        X, Y,
        linestyle = "-",
        color=color,
        linewidth=3,
        label=label_legend)
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})
    
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return


    
#------------------------------------------------------------------------------
#   Функция graph_lineplot_sns_np
#------------------------------------------------------------------------------

def graph_lineplot_sns(
    X, Y,
    Xmin_in=None, Xmax_in=None,
    Ymin_in=None, Ymax_in=None,
    color='orange',
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    y_label=None,
    label_fontsize=14, tick_fontsize=12,
    label_legend='', label_legend_fontsize=12,
    graph_size=(297/INCH, 210/INCH),
    file_name=None):
    
    sns.set_style("darkgrid")    # настройка цветовой гаммы
    fig, axes = plt.subplots(figsize=graph_size)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    if (Xmin_in != None) and (Xmax_in != None):
        Xmin=Xmin_in
        Xmax=Xmax_in
        axes.set_xlim(Xmin, Xmax)
            
    if (Ymin_in != None) and (Ymax_in != None):
        Ymin=Ymin_in
        Ymax=Ymax_in
        axes.set_ylim(Ymin, Ymax)        
        
    sns.lineplot(
        x=X, y=Y,
        palette=['orange'], color=color,
        linewidth=3,
        legend=True,
        label=label_legend,
        ax=axes)
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    #axes.axvline(x = 0, color = 'k', linewidth = 1)
    #axes.axhline(y = 0, color = 'k', linewidth = 1)
    axes.tick_params(labelsize = tick_fontsize)
    axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})
    
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return



#------------------------------------------------------------------------------
#   Функция graph_scatterplot_3D_mpl
#------------------------------------------------------------------------------

def graph_scatterplot_3D_mpl(
    X, Y, Z,
    elev=-160, azim=-60,
    Xmin_in=0, Xmax_in=3000,
    Ymin_in=-25, Ymax_in=25,
    Zmin_in=0, Zmax_in=20,
    color='orange',
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    y_label=None,
    z_label=None,
    label_fontsize=11, tick_fontsize=10,
    label_legend='', label_legend_fontsize=12,
    s=50,
    graph_size=(420/INCH, 297/INCH),
    file_name=None):
    
    sns.set_style("darkgrid")    # настройка цветовой гаммы
    fig = plt.figure(figsize=graph_size)
    axes = fig.add_subplot(111, projection='3d')
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    axes.set_title(title_axes, fontsize = title_axes_fontsize)
    
    if (Xmin_in != None) and (Xmax_in != None):
        Xmin=Xmin_in
        Xmax=Xmax_in
        axes.set_xlim(Xmin, Xmax)
            
    if (Ymin_in != None) and (Ymax_in != None):
        Ymin=Ymin_in
        Ymax=Ymax_in
        axes.set_ylim(Ymin, Ymax)        
    
    if (Zmin_in != None) and (Zmax_in != None):
        Zmin=Zmin_in
        Zmax=Zmax_in
        axes.set_zlim(Zmin, Zmax)  
    
    axes.scatter(
        X, Y, Z,
        label=label_legend,
        s=50,
        color=color)
    
    axes.set_xlabel(x_label, fontsize = label_fontsize)
    axes.set_ylabel(y_label, fontsize = label_fontsize)
    axes.set_zlabel(z_label, fontsize = label_fontsize)
    axes.view_init(elev=elev, azim=azim)   
    #axes.tick_params(labelsize = tick_fontsize)
    axes.legend(prop={'size': label_legend_fontsize})
        
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return   



#------------------------------------------------------------------------------
#   Функция graph_hist_boxplot_mpl
#
#   Принимает в качестве аргумента np.ndarray и строит совмещенный график - 
#   гистограмму и коробчатую диаграмму на одном рисунке (Figure) - с помощью 
#    библиотеки Mathplotlib.
#------------------------------------------------------------------------------    
    

def graph_hist_boxplot_mpl(
    X,
    Xmin=None, Xmax=None,
    bins_hist='auto',
    density_hist=False,
    title_figure=None, title_figure_fontsize=18,
    title_axes=None, title_axes_fontsize=16,
    x_label=None,
    label_fontsize=14, tick_fontsize=12, label_legend_fontsize=10,
    graph_size=(297/INCH, 210/INCH),
    file_name=None):
    
    X = np.array(X)
        
    if not(Xmin) and not(Xmax):
        Xmin=min(X)*0.99
        Xmax=max(X)*1.01
        
    # создание рисунка (Figure) и области рисования (Axes)
    fig = plt.figure(figsize=graph_size)
    ax1 = plt.subplot(2,1,1)    # для гистограммы
    ax2 = plt.subplot(2,1,2)    # для коробчатой диаграммы  
    
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    
    # заголовок области рисования (Axes)
    ax1.set_title(title_axes, fontsize = title_axes_fontsize)
    
    # выбор вида гистограммы (density=True/False - плотность/абс.частота) и параметры по оси OY
    ymax = max(np.histogram(X, bins=bins_hist, density=density_hist)[0])
    if density_hist:
        label_hist = "эмпирическая плотность распределения"
        ax1.set_ylabel('Относительная плотность', fontsize = label_fontsize)
        ax1.set_ylim(0, ymax*1.4)
    else:
        label_hist = "эмпирическая частота"
        ax1.set_ylabel('Абсолютная частота', fontsize = label_fontsize)
        ax1.set_ylim(0, ymax*1.4)

    # данные для графика плотности распределения
    nx = 100
    hx = (Xmax - Xmin)/(nx - 1)
    x1 = np.linspace(Xmin, Xmax, nx)
    #hx = 0.1; nx =  int(floor((Xmax - Xmin)/hx)+1)
    #x1 = np.linspace(Xmin, Xmax, nx)
    y1 = sps.norm.pdf(x1, X.mean(), X.std(ddof = 1))
    
    # рендеринг гистограммы
    if density_hist:
        ax1.hist(
            X,
            bins=bins_hist,    # выбор числа интервалов ('auto', 'fd', 'doane', '    ', 'stone', 'rice', 'sturges', 'sqrt')
            density=density_hist,
            histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
            orientation='vertical',   # 'vertical', 'horizontal'
            color = "#1f77b4",
            label=label_hist)
        ax1.plot(
            x1, y1,
            linestyle = "-",
            color = "r",
            linewidth = 2,
            label = 'теоретическая нормальная кривая')
    else:
        ax1.hist(
            X,
            bins=bins_hist,    # выбор числа интервалов ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
            density=density_hist,
            histtype='bar',    # 'bar', 'barstacked', 'step', 'stepfilled'
            orientation='vertical',   # 'vertical', 'horizontal'
            color = "#1f77b4",
            label=label_hist)    
        k = len(np.histogram(X, bins=bins_hist, density=density_hist)[0])
        y2 = y1*len(X)*(max(X)-min(X))/k
        ax1.plot(
            x1, y2,
            linestyle = "-",
            color = "r",
            linewidth = 2,
            label = 'теоретическая нормальная кривая')
    
    # Среднее значение, медиана, мода на графике
    ax1.axvline(
        X.mean(),
        color='magenta', label = 'среднее значение', linewidth = 2)
    ax1.axvline(
        np.median(X),
        color='orange', label = 'медиана', linewidth = 2)
    ax1.axvline(stat.mode(X),
        color='cyan', label = 'мода', linewidth = 2)
    
    ax1.set_xlim(Xmin, Xmax)
    ax1.tick_params(labelsize = tick_fontsize)
    ax1.grid(True)
    ax1.legend(fontsize = label_legend_fontsize)
    
    # рендеринг коробчатой диаграммы
    ax2.boxplot(
        X,
        vert=False,
        notch=False,
        widths=0.5,
        patch_artist=True)
    ax2.set_xlim(Xmin, Xmax)
    ax2.set_xlabel(x_label, fontsize = label_fontsize)
    ax2.tick_params(labelsize = tick_fontsize)
    ax2.grid(True)
        
    # отображение графика на экране и сохранение в файл
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return



#------------------------------------------------------------------------------
#   Функция Shapiro_Wilk_test
#   функция для обработки реализации теста Шапиро-Уилка
#------------------------------------------------------------------------------

def Shapiro_Wilk_test(data, p_level=0.95, DecPlace=5):
    a_level = 1 - p_level
    data = np.array(data)
    result = sci.stats.shapiro(data)
    s_calc = result.statistic    # расчетное значение статистики критерия
    a_calc = result.pvalue    # расчетный уровень значимости
    
    print(f"Расчетный уровень значимости: a_calc = {round(a_calc, DecPlace)}")
    print(f"Заданный уровень значимости: a_level = {round(a_level, DecPlace)}")
    
    if a_calc >= a_level:
        conclusion_ShW_test = f"Так как a_calc = {round(a_calc, DecPlace)} >= a_level = {round(a_level, DecPlace)}" + \
            ", то гипотеза о нормальности распределения по критерию Шапиро-Уилка ПРИНИМАЕТСЯ"
    else:
        conclusion_ShW_test = f"Так как a_calc = {round(a_calc, DecPlace)} < a_level = {round(a_level, DecPlace)}" + \
            ", то гипотеза о нормальности распределения по критерию Шапиро-Уилка ОТВЕРГАЕТСЯ"
    print(conclusion_ShW_test)



#------------------------------------------------------------------------------
#   Функция Epps_Pulley_test
#------------------------------------------------------------------------------    

def Epps_Pulley_test(data, p_level=0.95):
    a_level = 1 - p_level
    X = np.array(data)
    N = len(X)
    
    # аппроксимация предельных распределений статистики критерия
    сdf_beta_I = lambda x, a, b: sci.stats.beta.cdf(x, a, b, loc=0, scale=1)
    g_beta_III = lambda z, δ: δ*z / (1+(δ-1)*z)
    cdf_beta_III = lambda x, θ0, θ1, θ2, θ3, θ4: сdf_beta_I(g_beta_III((x - θ4)/θ3, θ2), θ0, θ1)
    # набор параметров распределения
    θ_1 = (1.8645, 2.5155, 5.8256, 0.9216, 0.0008)    # для 15 < n < 50
    θ_2 = (1.7669, 2.1668, 6.7594, 0.91, 0.0016)    # для n >= 50
    
    if N >= 8:
        # среднее арифметическое
        X_mean = X.mean()
        # центральный момент 2-го порядка
        m2 = np.var(X, ddof = 0)
        # расчетное значение статистики критерия
        A = sqrt(2) * np.sum([exp(-(X[i] - X_mean)**2 / (4*m2)) for i in range(N)])
        B = 2/N * np.sum(
            [np.sum([exp(-(X[j] - X[k])**2 / (2*m2)) for j in range(0, k)]) 
             for k in range(1, N)])
        s_calc_EP = 1 + N / sqrt(3) + B - A
        # табличное значение статистики критерия
        Tep_table_df = pd.read_csv(
            filepath_or_buffer='table/Epps_Pulley_test_table.csv',
            sep=';',
            index_col='n')
        p_level_dict = {
            0.9:   Tep_table_df.columns[0],
            0.95:  Tep_table_df.columns[1],
            0.975: Tep_table_df.columns[2],
            0.99:  Tep_table_df.columns[3]}
        f_lin = sci.interpolate.interp1d(Tep_table_df.index, Tep_table_df[p_level_dict[p_level]])
        s_table_EP = float(f_lin(N))
        # проверка гипотезы
        if 15 < N < 50:
            a_calc_EP = 1 - cdf_beta_III(s_calc_EP, θ_1[0], θ_1[1], θ_1[2], θ_1[3], θ_1[4])
            conclusion_EP = 'gaussian distribution' if a_calc_EP > a_level else 'not gaussian distribution'            
        elif N >= 50:
            a_calc_EP = 1 - cdf_beta_III(s_calc_EP, θ_2[0], θ_2[1], θ_2[2], θ_2[3], θ_2[4])
            conclusion_EP = 'gaussian distribution' if a_calc_EP > a_level else 'not gaussian distribution'            
        else:
            a_calc_EP = ''              
            conclusion_EP = 'gaussian distribution' if s_calc_EP <= s_table_EP else 'not gaussian distribution'            
                
    else:
        s_calc_EP = '-'
        s_table_EP = '-'
        a_calc_EP = '-'
        conclusion_EP = 'count less than 8'
    
    result = pd.DataFrame({
        'test': ('Epps-Pulley test'),
        'p_level': (p_level),
        'a_level': (a_level),
        'a_calc': (a_calc_EP),
        'a_calc >= a_level': (a_calc_EP >= a_level if N > 15 else '-'),
        'statistic': (s_calc_EP),
        'critical_value': (s_table_EP),
        'statistic < critical_value': (s_calc_EP < s_table_EP if N >= 8 else '-'),
        'conclusion': (conclusion_EP)},
        index=[1])  
        
    return result


#------------------------------------------------------------------------------
#   old Функция conjugacy_table_2x2_coefficient
#------------------------------------------------------------------------------

def conjugacy_table_2x2_independence_check (X, p_level=0.95):
    a_level = 1 - p_level
    data = np.array(X)  
    a = data[0][0]
    b = data[0][1]
    c = data[1][0]
    d = data[1][1]
    n = a + b + c + d
    
    u_p = sps.norm.ppf((1 + p_level)/2, 0, 1)    # табл.значение квантиля норм.распр.
    #print(u_p)
    
       
    # Коэффициент ассоциации Юла
    Q_calc = (a*d - b*c) / (a*d + b*c)    # расчетное значение коэффициента
    DQ = 1/4 * (1-Q_calc**2) * (1/a + 1/b + 1/c + 1/d)    # дисперсия
    Q_crit = u_p * sqrt(DQ)    # критическое значение коэффициента
    conclusion_Q = 'significant' if abs(Q_calc) >= Q_crit else 'not significant'        
    
    # Коэффициент коллигации Юла
    Y_calc = (sqrt(a*d) - sqrt(b*c)) / (sqrt(a*d) + sqrt(b*c))   # расчетное значение коэффициента
    DY = 1/16 * (1-Y_calc**2) * (1/a + 1/b + 1/c + 1/d)    # дисперсия
    Y_crit = u_p * sqrt(DY)    # критическое значение коэффициента
    conclusion_Y = 'significant' if abs(Y_calc) >= Y_crit else 'not significant'
    
    # Коэффициент контингенции Пирсона
    V_calc = (a*d - b*c) / sqrt((a + b)*(a + c)*(b + d)*(c + d))   # расчетное значение коэффициента
    DV = 1/n * (1-V_calc**2) + \
        1/n * (V_calc + 1/2 * V_calc**2) * ((a-d)**2 - (b-c)**2)/sqrt((a+b)*(a+c)*(b+d)*(c+d)) - \
            3/(4*n)*V_calc**2 * (((a+b-c-d)**2 / ((a+b)*(c+d))) - ((a+c-b-d)**2 / ((a+c)*(b+d))))    # дисперсия
    V_crit = u_p * sqrt(DV)    # критическое значение коэффициента
    conclusion_V = 'significant' if abs(V_calc) >= V_crit else 'not significant'
    
    # градация значений коэффициентов
    # шкала Эванса 1996 г. для психосоциальных исследований (https://medradiol.fmbafmbc.ru/journal_medradiol/abstracts/2019/6/12-24_Koterov_et_al.pdf)
    #strength of relationship
    check_Evans_scale = lambda r, measure_of_association: '-' if measure_of_association == 'not significant' else\
        'very weak' if abs(r) < 0.2 else \
            'weak' if abs(r) < 0.4 else \
                'moderate' if abs(r) < 0.6 else \
                    'strong' if abs(r) < 0.8 else \
                        'very strong'
    
    # Создадим DataFrame для сводки результатов
    coefficient = np.array([Q_calc, Y_calc, V_calc])
    critical_value = (Q_crit, Y_crit, V_crit)
    measure_of_association = (conclusion_Q, conclusion_Y, conclusion_V)
    strength_of_relationship = np.vectorize(check_Evans_scale)(coefficient, measure_of_association)
    
    func_of_measure = lambda value, measure_of_association: value if measure_of_association == 'significant' else '-'
    confidence_interval_min = func_of_measure(coefficient - critical_value, measure_of_association)
    confidence_interval_max = func_of_measure(coefficient + critical_value, measure_of_association)
                
    result = pd.DataFrame({
        'test': (
            'Yule’s Coefficient of Association (Q)',
            'Yule’s Coefficient of Colligation (Y)',
            "Pearson's contingency coefficient (V)"),
        'p_level': (p_level),
        'a_level': (a_level),
        'coefficient': coefficient,
        'critical_value': critical_value,
        '|coefficient| >= critical_value': (abs(coefficient) >= critical_value),
        'measure of association': (measure_of_association),
        'strength of relationship (Evans scale)': (strength_of_relationship),
        'confidence interval min': (confidence_interval_min),
        'confidence interval max': (confidence_interval_max)
        })
    return result

 

#------------------------------------------------------------------------------
#   old Функция conjugacy_table_IxJ_independence_check
#------------------------------------------------------------------------------

def conjugacy_table_IxJ_independence_check (X_in, p_level=0.95):
    a_level = 1 - p_level
    X = np.array(X_in)
    #print(X, '\n')
    result = []
    
    # параметры таблицы сопряженности
    N_rows = X.shape[0]
    N_cols = X.shape[1]
    N = X.size
    X_sum = np.sum(X)
    
    # проверка условия размерности таблицы 2х2
    check_condition_2x2 = True if (N_rows == 2 and N_cols == 2) else False
        
    # проверка условий применимости критерия хи-квадрат
    check_condition_chi2_1 = check_condition_chi2_2 = True
    note_condition_chi2_1 = note_condition_chi2_2 = ''
        
    if X_sum < 50:
        check_condition_chi2_1 = False
        note_condition_chi2_1 = 'sample size less than 50'
                    
    if check_condition_2x2:
        if np.size(np.where(X.ravel() < 5)):    # Функция np.ravel() преобразует матрицу в одномерный вектор
            check_condition_chi2_2 = False
            note_condition_chi2_2 = 'frequency less than 5'
    else:
        if np.size(np.where(X.ravel() < 3)):
            check_condition_chi2_2 = False
            note_condition_chi2_2 = 'frequency less than 3'
    
    # ожидаемые частоты и предельные суммы
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.expected_freq.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.margins.html
    
    
    # критерий хи-квадрат (без поправки Йетса)
    # https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
    s_calc_chi2 = '-'
    critical_value_chi2 = '-'
    significance_check_chi2_s = '-'
    a_calc_chi2 = '-'
    significance_check_chi2_a = '-'
    
    if check_condition_chi2_1 and check_condition_chi2_2:
        (s_calc_chi2, p_chi2, dof_chi2, ex_chi2) = sci.stats.chi2_contingency(X, correction=False)
        critical_value_chi2 = sci.stats.chi2.ppf(p_level, dof_chi2)
        significance_check_chi2_s = s_calc_chi2 >= critical_value_chi2
        a_calc_chi2 = p_chi2
        # альтернативный расчет: a_calc_chi2 = 1 - sci.stats.chi2.cdf(s_calc_chi2, dof_chi2)
        significance_check_chi2_a = a_calc_chi2 <= a_level
        conclusion_chi2 = 'categories are not independent' if significance_check_chi2_s else 'categories are independent'
    else:
        conclusion_chi2 = note_condition_chi2_1 + ' ' + note_condition_chi2_2
        
    # критерий хи-квадрат (с поправкой Йетса) (только для таблиц 2х2)
    # https://en.wikipedia.org/wiki/Yates%27s_correction_for_continuity
    # https://ru.wikipedia.org/wiki/%D0%9F%D0%BE%D0%BF%D1%80%D0%B0%D0%B2%D0%BA%D0%B0_%D0%99%D0%B5%D0%B9%D1%82%D1%81%D0%B0
    s_calc_chi2_Yates = '-'
    critical_value_chi2_Yates = '-'
    significance_check_chi2_s_Yates = '-'
    a_calc_chi2_Yates = '-'
    significance_check_chi2_a_Yates = '-'
    
    if check_condition_2x2:
        if check_condition_chi2_1 and check_condition_chi2_2:
            (s_calc_chi2_Yates, p_chi2_Yates, dof_chi2_Yates, ex_chi2_Yates) = sci.stats.chi2_contingency(X, correction=True)
            critical_value_chi2_Yates = sci.stats.chi2.ppf(p_level, dof_chi2_Yates)
            significance_check_chi2_s_Yates = s_calc_chi2_Yates >= critical_value_chi2_Yates
            a_calc_chi2_Yates = p_chi2_Yates
            significance_check_chi2_a_Yates = a_calc_chi2_Yates <= a_level
            conclusion_chi2_Yates = 'categories are not independent' if significance_check_chi2_s_Yates else 'categories are independent'
        else:
            conclusion_chi2_Yates = note_condition_chi2_1 + '   ' + note_condition_chi2_2                
    else:
        conclusion_chi2_Yates = 'not 2x2'
            
    # относительный риск
    # https://medstatistic.ru/methods/methods7.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.relative_risk.html#r092af754ff7d-1
    
    # точный критерий Фишера (Fisher's exact test) (односторонний)
    # https://en.wikipedia.org/wiki/Fisher's_exact_test
    # https://ru.wikipedia.org/wiki/%D0%A2%D0%BE%D1%87%D0%BD%D1%8B%D0%B9_%D1%82%D0%B5%D1%81%D1%82_%D0%A4%D0%B8%D1%88%D0%B5%D1%80%D0%B0
    # https://medstatistic.ru/methods/methods5.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
        
    # точный критерий Фишера (Fisher's exact test) (двусторонний)
    p_Fisher_two_sided = '-'
    significance_check_Fisher_two_sided = '-'
    a_calc_Fisher_two_sided = '-'
    significance_check_Fisher_two_sided = '-'
    conclusion_Fisher_two_sided = '-'
    
    if check_condition_2x2:
        (oddsr_Fisher_two_sided, p_Fisher_two_sided) = sci.stats.fisher_exact(X, alternative='two-sided')
        a_calc_Fisher_two_sided = p_Fisher_two_sided
        significance_check_Fisher_two_sided = a_calc_Fisher_two_sided <= a_level
        conclusion_Fisher_two_sided = 'categories are not independent' if significance_check_Fisher_two_sided else 'categories are independent'
    else:
        conclusion_Fisher_two_sided = 'not 2x2'
            
    # отношение шансов (odds ratio)
    # https://ru.wikipedia.org/wiki/%D0%9E%D1%82%D0%BD%D0%BE%D1%88%D0%B5%D0%BD%D0%B8%D0%B5_%D1%88%D0%B0%D0%BD%D1%81%D0%BE%D0%B2
    odds_ratio_calc = '-'
    significance_check_odds_ratio = '-'
    a_calc_odds_ratio = '-'
    significance_check_odds_ratio = '-'
    conclusion_odds_ratio = '-'
    
    if check_condition_2x2:
        odds_ratio_calc = oddsr_Fisher_two_sided
    else:
        conclusion_odds_ratio = 'not 2x2'
        
    # тест Барнарда 
    # https://en.wikipedia.org/wiki/Barnard's_test
    # https://wikidea.ru/wiki/Barnard%27s_test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.barnard_exact.html
    s_calc_Barnard = '-'
    critical_value_Barnard = '-'
    significance_check_Barnard = '-'
    a_calc_Barnard = '-'
    significance_check_Barnard = '-'
    
    if check_condition_2x2:
        res = sci.stats.barnard_exact(X, alternative='two-sided')
        s_calc_Barnard = res.statistic
        a_calc_Barnard = res.pvalue
        significance_check_Barnard = a_calc_Barnard <= a_level
        conclusion_Barnard = 'categories are not independent' if significance_check_Barnard else 'categories are independent'
    else:
        conclusion_Barnard = 'not 2x2'
        
    # тест Бошлу 
    # https://en.wikipedia.org/wiki/Boschloo%27s_test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boschloo_exact.html
    s_calc_Boschloo = '-'
    critical_value_Boschloo = '-'
    significance_check_Boschloo = '-'
    a_calc_Boschloo = '-'
    significance_check_Boschloo = '-'
    
    if check_condition_2x2:
        res = sci.stats.boschloo_exact(X, alternative='two-sided')
        s_calc_Boschloo = res.statistic
        a_calc_Boschloo = res.pvalue
        significance_check_Boschloo = a_calc_Boschloo <= a_level
        conclusion_Boschloo = 'categories are not independent' if significance_check_Boschloo else 'categories are independent'
    else:
        conclusion_Boschloo = 'not 2x2'        

    # заполним DataFrame для сводки результатов
    result = pd.DataFrame({
        'test': (
            'Chi-squared test',
            "Chi-squared test (with Yates's correction for 2x2)",
            "Fisher's exact test (two-sided)",
            "Odds ratio",
            "Barnard's exact test",
            "Boschloo's exact test"),
        'p_level': (p_level),
        'a_level': (a_level),
        'statistic': [s_calc_chi2, s_calc_chi2_Yates, '-', odds_ratio_calc, s_calc_Barnard, s_calc_Boschloo],
        'critical_value': [
            critical_value_chi2, 
            critical_value_chi2_Yates, 
            '-', '', 
            critical_value_Barnard, 
            critical_value_Boschloo],
        'statistic >= critical_value': [
            significance_check_chi2_s,
            significance_check_chi2_s_Yates,
            '-', '',
            s_calc_Barnard,
            s_calc_Boschloo],
        'a_calc': [a_calc_chi2, a_calc_chi2_Yates, a_calc_Fisher_two_sided, '', a_calc_Barnard, a_calc_Boschloo],
        'a_calc <= a_level': [
            significance_check_chi2_a,
            significance_check_chi2_a_Yates,
            significance_check_Fisher_two_sided,
            '', 
            significance_check_Barnard, 
            significance_check_Boschloo],
        'conclusion': [
            conclusion_chi2,
            conclusion_chi2_Yates,
            conclusion_Fisher_two_sided,
            '',
            conclusion_Barnard,
            conclusion_Boschloo]
        })

    return result



#------------------------------------------------------------------------------
#   old Функция graph_contingency_tables_bar_pd
#------------------------------------------------------------------------------

def graph_contingency_tables_bar_pd(
    data_pd,
    A_name = None, B_name = None, 
    graph_size=(210/INCH, 297/INCH/2),
    part_table=1/5,    # часть графика, выделяемая под таблицу с данными
    title_figure=None, title_figure_fontsize=16,
    file_name=None):
    
    # создание рисунка (Figure) и области рисования (Axes)
    fig = plt.figure(figsize=graph_size, constrained_layout=True)
    n_rows = int(1/part_table)
    gs = mpl.gridspec.GridSpec(nrows=n_rows, ncols=2, figure=fig)
    ax1_1 = fig.add_subplot(gs[0:n_rows-1, 0:1])
    ax1_2 = fig.add_subplot(gs[n_rows-1, 0:1])
    ax2_1 = fig.add_subplot(gs[0:n_rows-1, 1:])
    ax2_2 = fig.add_subplot(gs[n_rows-1, 1:])
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    ax1_1.set_title('Absolute values', fontsize=14)
    ax2_1.set_title('Relative values', fontsize=14)
    
    # столбчатая диаграмма с абсолютными значениями
    data_pd.plot.bar(
        stacked=True,
        rot=0,
        legend=True,
        ax=ax1_1)
    ax1_1.legend(loc='best', fontsize = 12, title=data_pd.columns.name)
    
    ax1_2.set_axis_off()
    table_1 = ax1_2.table(
        cellText=np.fliplr(np.array(data_pd)).T,
        rowLabels=data_pd.columns,
        #colLabels=df.index[0:3],
        cellLoc='center',
        loc='center')
    table_1.set_fontsize(12)
    table_1.scale(1, 3)
    
    # столбчатая диаграмма с относительными значениями
    data_relative = data_pd.copy()
    data_relative.iloc[:,0] = round(data_pd.iloc[:,0] / (data_pd.iloc[:,0] + data_pd.iloc[:,1]), 4)
    data_relative.iloc[:,1] = round(data_pd.iloc[:,1] / (data_pd.iloc[:,0] + data_pd.iloc[:,1]), 4)

    data_relative.plot.bar(
        stacked=True,
        rot=0,
        legend=True,
        color=['lightblue', 'wheat'],
        ax=ax2_1)
    ax2_1.legend(loc='best', fontsize = 12, title=data_pd.columns.name)
    
    ax2_2.set_axis_off()
    table_2 = ax2_2.table(
        cellText=np.fliplr(np.array(data_relative)).T,
        rowLabels=data_relative.columns,
        #colLabels=df.index[0:3],
        cellLoc='center',
        loc='center')
    table_2.set_fontsize(12)
    table_2.scale(1, 3)
    
    fig.tight_layout()
        
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return



#------------------------------------------------------------------------------
#   old Функция graph_contingency_tables_freqint_pd
#------------------------------------------------------------------------------

def graph_contingency_tables_bar_freqint_pd(
    data_pd,
    A_name = None, B_name = None, 
    graph_size=(297/INCH, 210/INCH/1.5),
    title_figure=None, title_figure_fontsize=16,
    file_name=None):
    
    # создание рисунка (Figure) и области рисования (Axes)
    fig = plt.figure(figsize=graph_size, constrained_layout=True)
    ax1 = plt.subplot(1,3,1)
    ax2 = plt.subplot(1,3,2)
    ax3 = plt.subplot(1,3,3)
    # заголовок рисунка (Figure)
    fig.suptitle(title_figure, fontsize = title_figure_fontsize)
    ax2.set_title('Relative values', fontsize=14)
    
    # столбчатая диаграмма с абсолютными значениями
    data_pd.plot.bar(
        stacked=True,
        rot=0,
        legend=True,
        ax=ax1)
    ax1.legend(loc='best', fontsize = 12, title=data_pd.columns.name)
    ax1.set_title('Absolute values', fontsize=14)
    
    # столбчатая диаграмма с относительными значениями
    data_relative = data_pd.copy()
    data_relative.iloc[:,0] = round(data_pd.iloc[:,0] / (data_pd.iloc[:,0] + data_pd.iloc[:,1]), 4)
    data_relative.iloc[:,1] = round(data_pd.iloc[:,1] / (data_pd.iloc[:,0] + data_pd.iloc[:,1]), 4)

    data_relative.plot.bar(
        stacked=True,
        rot=0,
        legend=True,
        color=['lightblue', 'wheat'],
        ax=ax2)
    ax2.legend(loc='best', fontsize = 12, title=data_pd.columns.name)
    
    # график взаимодействия частот
    sns.lineplot(
        data=data_pd,
        dashes=False,
        lw=3,
        #markers=['o','o'],
        markersize=10,
        ax=ax3)
    ax3.set_title('Graph of frequency interactions', fontsize=14)
    ax3.set_xticks(list(data_pd.index))
        
    fig.tight_layout()
        
    plt.show()
    if file_name:
        fig.savefig(file_name, orientation = "portrait", dpi = 300)
        
    return



#------------------------------------------------------------------------------
#   old Функция Mann_Whitney_test_trend_check
#
#   Проверяет гипотезу о наличии тренда (т.е. существенном различии двух частей
#   временного ряда) по критерию Манна-Уитни
#------------------------------------------------------------------------------    

def Mann_Whitney_test_trend_check(
        data1, data2,    # два части, на которые следует разбить исходный массив значений
        use_continuity=True,    # поправка на непрерывность
        alternative='two-sided',    # вид альтернативной гипотезы
        method='auto',    # метод расчета уровня значимости
        p_level=0.95,
        DecPlace=5):
    
    a_level = 1 - p_level
        
    result = sps.mannwhitneyu(
        data1, data2,
        use_continuity=use_continuity,
        alternative=alternative,
        method=method)
    s_calc = result.statistic    # расчетное значение статистики критерия
    a_calc = result.pvalue    # расчетный уровень значимости
    
    print(f"Расчетное значение статистики критерия: s_calc = {round(s_calc, DecPlace)}")
    print(f"Расчетный уровень значимости: a_calc = {round(a_calc, DecPlace)}")
    print(f"Заданный уровень значимости: a_level = {round(a_level, DecPlace)}")
          
    if a_calc >= a_level:
        conclusion_ShW_test = f"Так как a_calc = {round(a_calc, DecPlace)} >= a_level = {round(a_level, DecPlace)}" + \
            ", то нулевая гипотеза об отсутствии сдвига в выборках ПРИНИМАЕТСЯ, т.е. сдвиг ОТСУТСТВУЕТ"
    else:
        conclusion_ShW_test = f"Так как a_calc = {round(a_calc, DecPlace)} < a_level = {round(a_level, DecPlace)}" + \
            ", то нулевая гипотеза об отсутствии сдвига в выборках ОТКЛОНЯЕТСЯ, т.е. сдвиг ПРИСУТСТВУЕТ"
    print(conclusion_ShW_test)
    return    

    