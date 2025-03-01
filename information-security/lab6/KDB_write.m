function [rgbImgtxt, s, coords] = KDB_write(rgbImg, bin_txt)
% Запись сообщения в изображение по алгоритму Куттера-Джордана-Боссена
% В данном методе шифрования 1 бит кодируется в 1 пикселе изображения, 
% путем изменения яркости синего канала в зависимости от значения, записанного в данном бите 

L = 0.1; % Коэфициент, влияющий на яркость пикселя со встроенными данными
r = 5; % Количество встраиваний каждого бита сообщения
s = size(bin_txt);
simg = size(rgbImg);
coordy = randi([4, simg(1) - 3], s(1), s(2), r);
coordx = randi([4, simg(2) - 3], s(1), s(2), r);
coords = cat(3, coordy, coordx);
rgbImgtxt = rgbImg;

for i = 1 : s(1)
    for j = 1 : s(2) 
        for k = 1 : r
            % Яркость пикселя
            Y = (0.298 * rgbImg(coords(i, j, k), coords(i, j, k + 3), 1)) ...
                    + (0.586*rgbImg(coords(i, j, k), coords(i, j, k + 3), 2)) ...
                    + (0.114*rgbImg(coords(i, j, k), coords(i, j, k + 3), 3));
            if (Y == 0)
                Y = 5 / L; 
            end
        
            % Условный оператор определяющий в какую сторону необходимо
            % изменить цвет пикселя
        
            if (bin_txt(i, j) == 1)
               rgbImgtxt(coords(i, j, k), coords(i, j, k + 3), 3) = double(rgbImg(coords(i, j, k), coords(i, j, k + 3), 3) + L * Y);
            else
               rgbImgtxt(coords(i, j, k), coords(i, j, k + 3), 3) = double(rgbImg(coords(i, j, k), coords(i, j, k + 3), 3) - L * Y);
            end
            
            % корректируем границы [0 255]
            if (rgbImgtxt(coords(i, j, k), coords(i, j, k + 3), 3) > 255)
               rgbImgtxt(coords(i, j, k), coords(i, j, k + 3), 3) = 255;
            end
            if (rgbImgtxt(coords(i, j, k), coords(i, j, k + 3), 3) < 0)
               rgbImgtxt(coords(i, j, k), coords(i, j, k + 3), 3) = 0;
            end
        end
    end
end
end

