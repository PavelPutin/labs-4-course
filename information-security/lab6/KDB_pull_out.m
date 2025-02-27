function [txt_new] = KDB_pull_out(rgbImgtxt,  s,  coords)
% Извлечение сообщения из изображения по алгоритму Куттера-Джордана-Боссена
 
% Извлечение сообщения происходит путем прогнозирования яркости пикселя по 
% яркости соседних пикселей. Значение бита сообщения определяется в 
% зависимости от того больше или меньше яркость пикселя от спрогнозированной. 
rgbImgprog = rgbImgtxt;
sigma = 3;
r = 5;
txt_new_bin = zeros(s(1), s(2))
for i = 1 : s(1)
    for j = 1 : s(2) 
        kat = zeros(1, r);
        for k = 1 : r
            % Рассчетная яркость пикселя
            rgbImgprog = (double(sum(rgbImgtxt(coords(i, j, k) - sigma:coords(i, j, k) + sigma, coords(i, j, k + 3), 3))) ...
                        + double(sum(rgbImgtxt(coords(i, j, k), coords(i, j, k + 3) - sigma:coords(i, j, k + 3) + sigma, 3))) ...
                        - 2 * double(rgbImgtxt(coords(i, j, k), coords(i, j, k + 3), 3))) ...
                    / (4 * sigma);
            % Условный оператор определяющий в какую сторону необходимо
            % изменить цвет пикселя
            del = double(rgbImgtxt(coords(i, j, k), coords(i, j, k + 3), 3)) - rgbImgprog;
            if del == 0 && rgbImgprog == 255
        	    del = 0.5;
            end
            
            if del == 0 && rgbImgprog == 0
       	     del =  -0.5;
            end
            
            if del > 0
               kat(k) = 1;
            else
               kat(k) = 0;
            end
        end
        txt_new_bin(i, j) = round(sum(kat) / r);
    end
end
txt_new_doub = bin2dec(num2str(txt_new_bin));
s = size(txt_new_doub);
txt_new = char(reshape(txt_new_doub, s(2), s(1)));
end

