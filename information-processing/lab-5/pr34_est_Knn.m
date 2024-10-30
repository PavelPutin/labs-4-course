%Файл pr34_est_Knn. Оценка плотности распределения вероятностей методом k-ближайших соседей
clear all; close all;

%% Одномерный случай

% 1. Исходные данные
n=1; %n-размерность вектора наблюдений
N=1000; %количество используемых для оценк и векторов
gm=0.5;
k=round(N^gm); %k - число ближайших соседей

%2. Генерация обучающей выборки и отсчётов эталонной плотности для одномерного случая
x0=-3:0.05:3; %задание сетки отсчётов (области значений СВ, для которой визуализируется оценка)
p=zeros(1,length(x0)); 

%2.Генерация обучающей выборки и отсчётов эталонной плотности для одномерного случая
x0=-3:0.05:3; %задание сетки отсчётов (области значений СВ, для которой визуализируется оценка)
p=zeros(1,length(x0)); 

% Раскомментите один из подпунктов

% %2.а) Гауссовское распределение СВ с параметрами: m=0, D=1;
% XN=randn(1,N);
% p=exp(-x0.^2/2)/sqrt(2*pi);%вид оцениваемой плотности (гауссовская)

% %2.б) Равномерное распределение СВ с параметрами a=0, b=1;
% XN=rand(1,N); 
% ind1=logical(x0>0 & x0<=1);
% p(ind1)=1;%вид оцениваемой плотности (равномерная)

% 2.в) Показательное распределение СВ с параметром b=1;
XN=-log(rand(1,N));
ind1=logical(x0>0);
p(ind1)=exp(-x0(ind1));%вид оцениваемой плотности  (показательная)

% 3. Оценка плотности по методу К ближайших соседей
p_=vknn(x0,XN,k);%оценка плотности

% 4. Отображение графиков плотности распределения и ее оценки
figure; grid on; hold on;
axis([min(x0) max(x0) 0 max(p)+0.2]);%установка границ поля графика по осям
plot(x0,p,'-b',x0,p_,'-+k');
title('Плотность распределения и ее оценка','FontName','Courier','FontSize',14);
xlabel('x','FontName','Courier'); ylabel('p','FontName','Courier');
strv1='N= '; strv2=num2str(N); strv3='  k= '; strv4=num2str(k);
text(-2.5, 0.97*max(p), [strv1, strv2, strv3, strv4], 'HorizontalAlignment','left',...
      'BackgroundColor',[.8 .8 .8],'FontSize',12);
legend('p ','p~'); hold off;

%% Двумерный случай
% 1. Исходные данные
n=2; %n-размерность вектора наблюдений
N=2000; %количество используемых для оценк и векторов
gm=0.5;
k=round(N^gm); %k - число ближайших соседей

% 2.Генерация отсчетов эталонной плотности (в виде смеси гауссиан) для двумерного случая
% Параметры распределения смеси гауссовских случайных векторов;
M=3; %количество компонентов в смеси
ps=[0.2,0.2,0.6]; %вероятности появления СВ различных типов в смеси
% Расчет матрицы ковариаций ГСВ смеси
D=0.2; ro=-log(0.7); %дисперсия и коэффициент корреляции cоседних элементов 
% Расположение математических ожиданий компонентов смеси
m1=[0;0]; m2=[1;0]; m3=[0;1]; m=[m1,m2,m3];
% Ковариационная матрица компонентов смеси
for i=1:n, for j=1:n,
        C(i,j)=D*exp(-ro*abs(i-j));
end;end;
x1=[-2:0.1:3]; x2= [-2:0.1:3]; %области значений СВ, для которой визуализируется оценка
[X1,X2]=meshgrid(x1,x2); x=[X1(:) X2(:)]'; % матрицы Х и Y координат отсчётов
% Значения эталонной плотности
p=ps(1)*mvnpdf(x',m1',C)+ps(2)*mvnpdf(x',m2',C)+ps(3)*mvnpdf(x',m3',C);
pi=reshape(p,length(x1),length(x2));%матрица значений плотности распределения

% 3. Отображение графика плотности распределения
figure; grid on; hold on;
caxis([min(pi(:))-0.5*range(pi(:)),max(pi(:))+0.5*range(pi(:))]);%установка цвета 
axis([min(x1) max(x1)  min(x1) max(x1) 0 max(pi(:))+0.1]);%границы поля графика 
surf(x1,x2,pi);
title('Распределение смеси гауссовских векторов','FontName','Courier','FontSize',14);
xlabel('x1','FontName','Courier'); ylabel('x2','FontName','Courier'); zlabel('pi');
strv1='N= '; strv2=num2str(N);strv3=' k= '; strv4=num2str(k);
text(1, -0.5, 0.5*max(pi(:)),[strv1,strv2,strv3,strv4], 'HorizontalAlignment','left',...
    'BackgroundColor',[.8 .8 .8],'FontSize',12);
hold off;

% 4. Обучающая выборка
XN=zeros(n,N);
for i=1:N,%генерация обучающей выборки 
     u=rand;
     %индекс принадлежности к компоненте смеси
     if u<ps(1), t=1; elseif u<ps(1)+ps(2), t=2; else t=3; end;
     XN(:,i)=randncor(n,1,C)+m(:,t);
end;

% 5. Оценка плотности по методу К ближайших соседей
p_=vknn(x,XN,k);%оценка плотности
%матрицы значений координат случайного вектора
pv=reshape(p_,length(x1),length(x2));%матрица значений оценки плотности распределения

% 6. Отображение графика оценки плотности распределния
figure; grid on; hold on;
caxis([min(pv(:))-0.5*range(pv(:)),max(pv(:))+0.5*range(pv(:))]);%установка цветовой гаммы
axis([min(x1),max(x1),min(x1),max(x1),0,max(pi(:))+0.1]);%установка границ роля графика по осям
surf(x1,x2,pv);
title(' Оценка плотности распределения смеси ','FontName','Courier','FontSize',14);
xlabel('x1','FontName','Courier'); ylabel('x2','FontName','Courier'); zlabel('p~');
text(1, -0.5, 0.5*max(pv(:)),[strv1,strv2,strv3,strv4], 'HorizontalAlignment','left',...
    'BackgroundColor',[.8 .8 .8],'FontSize',12);
hold off;
% 7. Визуализация линий постоянного уровня плотности распределения и ее оценки
figure;hold on;
[Cv,h]=contour(x1,x2,pi,[0.001 0.01 0.5*max(pi(:))]);
clabel(Cv,h);
title('Линии постоянного уровня плотности','FontName','Courier','FontSize',14);
xlabel('x1','FontName','Courier'); ylabel('x2','FontName','Courier'); 
text(0, 0, [strv1,strv2,strv3,strv4], 'HorizontalAlignment','left',...
    'BackgroundColor',[.8 .8 .8],'FontSize',12);
hold off;
figure;hold on;
[Cv,h]=contour(x1,x2,pv,[0.001 0.01 0.5*max(pv(:))]);
clabel(Cv,h);
title('Линии постоянного уровня оценки плотности','FontName','Courier','FontSize',14);
xlabel('x1','FontName','Courier'); ylabel('x2','FontName','Courier'); 
text(0, 0, [strv1,strv2,strv3,strv4], 'HorizontalAlignment','left',...
    'BackgroundColor',[.8 .8 .8],'FontSize',12);
hold off;
