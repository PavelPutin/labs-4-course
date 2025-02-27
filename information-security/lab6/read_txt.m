function [txt, bin_txt] = read_txt()
% Функция предназначена для считывания информации из текстового файла
% Функция открывает файл в режиме чтения и записывает его содержимое в
% переменную txt, которая возвращается вызвавшему ее скрипту

txt_r = '';
txt = '';
text1 = fopen('message.txt','r'); 

while feof(text1) == 0
    line = fgetl(text1);
    txt_r = char(txt_r, line);
end

fclose(text1);

txt(1,:) = txt_r(find((double(txt_r)>1040) + (double(txt_r)<1103)));
s = size(txt);
txt = txt(2:2:s(2));
bin_txt = dec2bin(double(txt)) - '0';
end

