%%% Вычисление выборочных характеристик гауссовской случайной величины (ГСВ)
clear all
close all

%% 1. Задание исходных данных
% Параметры распределения
mu = 1;
sig = 0.5;
% Параметры генерации
n = 12;     % размерность равномерной случайной величины (РСВ)
N = 1000;   % число реализаций

%% 2. Вычисление значений статистических характеристик ГСВ
m = mu;     % мат. ожидание
d = sig^2;  % дисперсия
% Функция для вычисления значений плотности распределения
p = @(x) 1 / (sig * sqrt(2*pi)) * exp(-(x - mu).^2 / (2 * sig^2));

%% 3. Генерация реализаций случайной величины
% Генерация реализаций стандартной РСВ
alf = rand(n, N);   % матрица из N столбцов по n элементов
% Генерация реализаций ГСВ (суммирование выполняется по столбцам матрицы alf)
x = sig * (sum(alf) - 6) + mu;

%% 4. Вычисление выборочных характеристик
M = mean(x);        % выборочное среднее
D = var(x);         % выборочная дисперсия
% Вывод значений теоретических и выборочных характеристик
disp('Среднее значение (теоретическое)'); 
disp(m);
disp('Среднее значение (выборочное)');
disp(M);
disp('Дисперсия (теоретическая)');
disp(d);
disp('Дисперсия (выборочная)');
disp(D);

%% 5. Формирование вариационного ряда и интервалов гистограмм
% Построение вариационного ряда
X = sort(x);        % упорядочивание реализаций ГСВ по возрастанию
% Вычисление размаха
RN = X(end) - X(1);
% Вывод значения размаха
disp("Размах");
disp(RN);
% Определение границ интервалов
h = RN / (1 + 3.2 * log2(N));   % ширина интервала
H = 1 + fix(RN / h);            % число интервалов гистограмм   
X0 = X(1) - h/2;                % нижняя граница
Xc = X(1) + h .* (0 : H - 1);   % центры интервалов
Xi = [ X0, Xc + h/2 ];          % все границы интервалов

%% 6. Построение интервальной эмпирической функции распределения
% Вычисление значений функции распределения
F = zeros(1, H);
for i = 1 : H
% Интеграл по плотности от нижней границы до верхней границы i-го интервала
    F(i) = integral(p, X0, Xi(i + 1));
end
% Визуализация интервальной эмпирической функции распределения
figure;         % создание нового графического окна
Fx = histogram(x, Xi, 'Normalization', 'cdf');
Fv = Fx.Values; % получение значений в ячейках
% Отрисовка теоретической функции распределения
hold on;        % включение режима дорисовки
plot(Xc, F);    % визуализация функции распределения
legend('Инт. эмпирическая ф-я распред-я', 'Теоретическая ф-я распред-я');
% Вычисление средней разности по центральным точкам интервалов
err_F = mean(abs(F - Fv));
disp('Средняя разность теоретической и эмпирической функций');
disp(err_F);

%% 7. Построение эмпирической плотности распределения
% Вычисление значений плотности распределения
f = zeros(1, H);
for i = 1 : H
    % Значение плотности в центральной точке i-го интервала
    f(i) = p(Xc(i));
end
% Визуализация нормированной гистограммы
figure;         % создание нового графического окна
fx = histogram(x, Xi, 'Normalization', 'pdf');
fv = fx.Values; % получение значений в ячейках гистограммы
% Отрисовка плотности распределения
hold on;        % включение режима дорисовки
plot(Xc, f);    % визуализация плотности распределения
legend('Нормированная гистограмма', 'Плотность распределения');
% Вычисление средней разности по центральным точкам интервалов
err_f = mean(abs(f - fv));
disp('Средняя разность теоретической и эмпирической плотностей');
disp(err_f);