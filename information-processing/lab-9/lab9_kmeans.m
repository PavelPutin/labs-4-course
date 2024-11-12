% Файл pr72_kmeans.Программа для тестирования алгоритма кластеризации 
% на основе метода K-means 
close all;
%% 1. Исходные данные для генерации образов M порождающих классов
n=2; M=5;%размерность признакового пространства и число классов 
% L - количество компонентов смеси в каждом классе 
% dm - параметр, определяющий среднюю степень пересечения компонентов смесей
% romin, romax - границы значений коэффициента корреляции для задания матриц ковариации
L=ones(1,M);%каждый класс порождается одним гауссовским распределением 
dm=4; romin=-0.9; romax=0.9; 
% Веса, математические ожидания, дисперсии и коэффициенты корреляции компонентов смесей
ps=cell(1,M); mM=cell(1,M); D=cell(1,M); ro=cell(1,M);
for i=1:M
    ps{i}=ones(1,L(i))/L(i); D{i}=ones(1,L(i));  ro{i}=romin+(romax-romin)*rand(1,L(i)); 
end
mM{1}=[0;0]; mM{2}=[0;dm]; mM{3}=[dm;0]; mM{4}=[dm;dm];  mM{5}=[-dm;-dm];     
Ni=50; NN=[Ni,Ni,Ni,Ni,Ni]; N=sum(NN); % объемы тестирующих данных 
%% 2. Тестирование алгоритма
options=statset('Display','final','MaxIter',100,'TolFun',1e-6);
X=gen(n,M,NN,L,ps,mM,D,ro,0);
Nmi=0; Ns=zeros(1,M); XN=zeros(N,n);
for i=1:M, Nma=Nmi+NN(i); Ns(i)=Nma; XN(Nmi+1:Nma,:) =X{i}'; Nmi=Nma; end;
[idx,ctrs,sumd] = kmeans(XN,M,'Distance','sqeuclidean','replicates',5,'Options',options);
% idx - индекс принадлежности данных каждому кластеру
% ctrx - центры каждомго кластера
% sumd - сумма квадратов эвклидова расстояния точек внутри каждого кластера до центра
figure(1); silhouette(XN,idx); %отображение силуэта
%% 3. Оценка ошибок, визуализация  тестовых данных и ошибочных решений
[ercl,idxn,prM] = erclust(M,NN,idx);%оценка ошибок
disp('Индекс качества кластеризации и частость ошибок (sqeuclidean)'); disp([prM,ercl]);

figure; grid on; hold on;
title('sqeuclidean')
plot(XN(1:Ns(1),1),XN(1:Ns(1),2),'ko','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(1)+1:Ns(2),1),XN(Ns(1)+1:Ns(2),2),'r^','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(2)+1:Ns(3),1),XN(Ns(2)+1:Ns(3),2),'b+','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(3)+1:Ns(4),1),XN(Ns(3)+1:Ns(4),2),'m<','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(4)+1:Ns(5),1),XN(Ns(4)+1:Ns(5),2),'g*','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==1,1),XN(idxn==1,2),'ko','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==2,1),XN(idxn==2,2),'r^','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==3,1),XN(idxn==3,2),'b+','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==4,1),XN(idxn==4,2),'m<','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==5,1),XN(idxn==5,2),'g*','MarkerSize',10,'LineWidth',1);
plot(ctrs(:,1),ctrs(:,2),'k*','MarkerSize',14,'LineWidth',2);
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5'); hold off;

% + Добавить кластеризацию с другими метриками
[idx,ctrs,sumd] = kmeans(XN,M,'Distance','cityblock','replicates',5,'Options',options);
figure(1); silhouette(XN,idx); %отображение силуэта
[ercl,idxn,prM] = erclust(M,NN,idx);%оценка ошибок
disp('Индекс качества кластеризации и частость ошибок (cityblock)'); disp([prM,ercl]);

figure; grid on; hold on;
title('cityblock')
plot(XN(1:Ns(1),1),XN(1:Ns(1),2),'ko','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(1)+1:Ns(2),1),XN(Ns(1)+1:Ns(2),2),'r^','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(2)+1:Ns(3),1),XN(Ns(2)+1:Ns(3),2),'b+','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(3)+1:Ns(4),1),XN(Ns(3)+1:Ns(4),2),'m<','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(4)+1:Ns(5),1),XN(Ns(4)+1:Ns(5),2),'g*','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==1,1),XN(idxn==1,2),'ko','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==2,1),XN(idxn==2,2),'r^','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==3,1),XN(idxn==3,2),'b+','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==4,1),XN(idxn==4,2),'m<','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==5,1),XN(idxn==5,2),'g*','MarkerSize',10,'LineWidth',1);
plot(ctrs(:,1),ctrs(:,2),'k*','MarkerSize',14,'LineWidth',2);
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5'); hold off;

% + Добавить кластеризацию с другими метриками
[idx,ctrs,sumd] = kmeans(XN,M,'Distance','correlation','replicates',5,'Options',options);
figure(1); silhouette(XN,idx); % отображение силуэта
[ercl,idxn,prM] = erclust(M,NN,idx); % оценка ошибок
disp('Индекс качества кластеризации и частость ошибок (correlation)'); disp([prM,ercl]);

figure; grid on; hold on;
title('correlation')
plot(XN(1:Ns(1),1),XN(1:Ns(1),2),'ko','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(1)+1:Ns(2),1),XN(Ns(1)+1:Ns(2),2),'r^','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(2)+1:Ns(3),1),XN(Ns(2)+1:Ns(3),2),'b+','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(3)+1:Ns(4),1),XN(Ns(3)+1:Ns(4),2),'m<','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(4)+1:Ns(5),1),XN(Ns(4)+1:Ns(5),2),'g*','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==1,1),XN(idxn==1,2),'ko','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==2,1),XN(idxn==2,2),'r^','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==3,1),XN(idxn==3,2),'b+','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==4,1),XN(idxn==4,2),'m<','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==5,1),XN(idxn==5,2),'g*','MarkerSize',10,'LineWidth',1);
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5'); hold off;

% + Добавить кластеризацию с другими метриками
[idx,ctrs,sumd] = kmeans(XN,M,'Distance','cosine','replicates',5,'Options',options);
figure(1); silhouette(XN,idx); % отображение силуэта
[ercl,idxn,prM] = erclust(M,NN,idx); % оценка ошибок
disp('Индекс качества кластеризации и частость ошибок (cosine)'); disp([prM,ercl]);

% Сделить чтоб Индекс качества кластеризации равнялся 1 и частость ошибок
% была минимальной

figure; grid on; hold on;
title('cosine')
plot(XN(1:Ns(1),1),XN(1:Ns(1),2),'ko','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(1)+1:Ns(2),1),XN(Ns(1)+1:Ns(2),2),'r^','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(2)+1:Ns(3),1),XN(Ns(2)+1:Ns(3),2),'b+','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(3)+1:Ns(4),1),XN(Ns(3)+1:Ns(4),2),'m<','MarkerSize',10,'LineWidth',1);
plot(XN(Ns(4)+1:Ns(5),1),XN(Ns(4)+1:Ns(5),2),'g*','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==1,1),XN(idxn==1,2),'ko','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==2,1),XN(idxn==2,2),'r^','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==3,1),XN(idxn==3,2),'b+','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==4,1),XN(idxn==4,2),'m<','MarkerSize',10,'LineWidth',1);
plot(XN(idxn==5,1),XN(idxn==5,2),'g*','MarkerSize',10,'LineWidth',1);
plot(ctrs(:,1),ctrs(:,2),'k*','MarkerSize',14,'LineWidth',2);
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5'); hold off;