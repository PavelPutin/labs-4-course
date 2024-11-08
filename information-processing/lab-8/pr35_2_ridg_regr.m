%Файл pr35_2_ridg_regr. Построение гребневой регрессии
clear all; close all;
%1.Задание исходных данных
N=100;%объем обучающей выборки 
n=2;%размерность вектора входной переменной
K=10;%число шагов цикла изменения параметра регуляризации
eps=0.0000001;%параметр степени обусловловленности матрицы ковариации
D=0.1; %дисперсия ошибки измерения выходной переменной 
delt=3*sqrt(D*N); %пороговый уровень невязки
C=eye(n);%матрица ковариации входного вектора
C(n,:)=(1-eps)*ones(1,n); C(:,n)=(1-eps)*ones(1,n); C(n,n)=n-1;
xmin=-3; xmax=3; %границы области определения входных переменных
A=[-1,(1:n)]';%вектор оцениваемых параметров регрессии
%2.Генерация обучающей выборки данных и выборки новых данных
[XN,m]=randncor(n,N,C);
X=[ones(N,1),XN']; YN=X*A+sqrt(D)*randn(N,1);
XtX=X'*X;
betta1=max(eig(XtX))/1.0e+03;%рекомендуемое значение параметра регуляризации
rc=cond(C); rx=cond(XtX); rxb=cond(XtX+betta1*eye(n+1));
disp(['Числа обусловленности матриц: С, XtX,XtX+bI']);
disp([rc,rx,rxb]);

%Генерация новых данных
XNn=xmin+(xmax-xmin)*rand(n,N); 
Xn=[ones(N,1),XNn']; YNn=Xn*A+sqrt(D)*randn(N,1);
Mb=zeros(K,1); Rb=zeros(K,1); Rbn=zeros(K,1); Ea=zeros(K,1);fm=zeros(1,K);
betta=0.0000001;%начальное значение параметра регуляризации
%3.Восстановление регрессии при разных значениях параметра регуляризации
for k=1:K, %цикл по значениям параметра регуляризации
    betta=betta*10; Mb(k)=log10(betta);
    %Определение коэффициентов гребневой регрессии
    a_=(X'*X+betta*eye(n+1))\(X'*YN);
    %Расчет невязок и СКО оценивания коэффициентов модели
    Rb(k)=sqrt((X*a_-YN)'*(X*a_-YN));
    Rbn(k)=sqrt((Xn*a_-YNn)'*(Xn*a_-YNn));
    Ea(k)=sqrt((A-a_)'*(A-a_));
end;%цикл по k
%4.Визуализация результатов
%Визуализация входных данных в 3D
if n==3,
     figure(1);set(gca,'FontSize',12); grid on; hold on;
     pg3=plot3(XN(1,:),XN(2,:),XN(3,:),'ro'); set(pg3,'LineWidth',1.25);
     xlabel('x1','FontName','Courier');ylabel('x2','FontName','Courier');zlabel('x3','FontName','Courier');
     title('Пространственная локализация входных данных','FontName','Courier'); hold off;   
 end;
%Построение графиков для выбора параметра регуляризации 
figure(2);set(gca,'FontSize',12); grid on; hold on;
%ms=max([max(Rb),max(Rbn),max(Ea),delt])+1;
ms=3*delt;
axis([min(Mb) max(Mb) 0 ms]);
pg=plot(Mb,Rb,'-b^',Mb,Rbn,'--b^',Mb,Ea,'--r+',Mb,delt*ones(1,K),'--k');
set(pg,'LineWidth',1.25);
bl=log10(betta1);%рекомендуемое значение betta на оси OX
line('Xdata',[bl,bl],'Ydata',[0,ms],'LineWidth',1.5);
title('Дисперсионный анализ полученной регресии ','FontName','Courier');
xlabel('Значение параметра регуляризации (betta)','FontName','Courier')
str1='N='; str2=num2str(N);str3=' eps='; str4=num2str(eps);str5=' D='; str6=num2str(D);
text(Mb(1)+1,ms-delt,[str1, str2, str3, str4, str5, str6], 'HorizontalAlignment','left','BackgroundColor',[.8 .8 .8],'FontSize',12);
legend('Rb','Rbn','Ea','delt','bl');hold off;   
