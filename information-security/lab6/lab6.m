close all;
clear;
clc;

% считывание исходного RGB-изображение 
rgbImg = imread('good morning.bmp');

% вывод на экран исходного изображения
figure(1);
imshow(rgbImg);
title('Считанный файл *.bmp в режиме RGB');

% считывание текста из файла
[txt, bin_txt] = read_txt();
disp(txt);

% запись считанной текстовой информации в синий канал изображния
[rgbImgtxt, s, coords] = KDB_write(rgbImg, bin_txt);

figure(2)
imshow( rgbImgtxt );
title('Изображение со встроенной информацией');

% Извлечение информации из синего канала изображения

[txt_new] = KDB_pull_out(rgbImgtxt, s, coords);
disp(txt_new);
