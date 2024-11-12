%Файл pr73_hierarchical. Программа для тестирования алгоритма 
%иерархической агломеративной кластеризации 
clear all; close all;
%1.Исходные данные для генерации образов M порождающих классов
n=2; M=4;%размерность признакового пространства и число классов 
%L - количество компонентов смеси в каждом классе 
%dm - параметр, определяющий среднюю степень пересечения компонентов смесей
%romin, romax -  границы значений коэффициента корреляции для задания матриц ковариации
L=ones(1,M);%каждый класс порождается одним гауссовским распределением 
dm=3; romin=-0.9; romax=0.9; 
%Веса, математические ожидания, дисперсии и коэффициенты корреляции компонентов смесей
ps=cell(1,M); mM=cell(1,M); D=cell(1,M); ro=cell(1,M);
for i=1:M, 
    ps{i}=ones(1,L(i))/L(i); D{i}=ones(1,L(i));  ro{i}=romin+(romax-romin)*rand(1,L(i)); 
end;
mM{1}=[0;0]; mM{2}=[0;dm]; mM{3}=[dm;0]; mM{4}=[dm;dm];      
Ni=50; NN=[Ni,Ni,Ni,Ni]; N=sum(NN);%объемы тестирующих данных 
%2.Тестирование алгоритма 
X=gen(n,M,NN,L,ps,mM,D,ro,0);
Nmi=0; Ns=zeros(1,M); XN=zeros(N,n);
for i=1:M, Nma=Nmi+NN(i); Ns(i)=Nma; XN(Nmi+1:Nma,:) =X{i}'; Nmi=Nma; end;
Qn = linkage(XN,'ward','euclidean');%построение дендрограммы
figure(1); dendrogram(Qn);             %отображение дендрограммы
idx = cluster(Qn,'maxclust',M); %выполнение кластеризации на основе дендрограммы
%idx = clusterdata(XN,'maxclust',M,'linkage','ward'); %выполнение кластеризации на основе данных
%idx-индекс принадлежности данных каждому кластеру
%3.Оценка ошибок, визуализация  тестовых данных и ошибочных решений
[ercl,idxn,prM] = erclust(M,NN,idx);%оценка ошибок
disp('Индекс качества кластеризации и частость ошибок'); disp([prM,ercl]);
figure(2); grid on; hold on;
plot(XN(1:Ns(1),1),XN(1:Ns(1),2),'ko','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(1)+1:Ns(2),1),XN(Ns(1)+1:Ns(2),2),'r^','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(2)+1:Ns(3),1),XN(Ns(2)+1:Ns(3),2),'b+','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(3)+1:Ns(4),1),XN(Ns(3)+1:Ns(4),2),'m<','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==1,1),XN(idxn==1,2),'ko','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==2,1),XN(idxn==2,2),'r^','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==3,1),XN(idxn==3,2),'b+','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==4,1),XN(idxn==4,2),'m<','MarkerSize',10,'LineWidth',1);
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4',1); hold off;
