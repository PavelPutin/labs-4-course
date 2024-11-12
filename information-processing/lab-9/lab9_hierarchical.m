%���� pr73_hierarchical. ��������� ��� ������������ ��������� 
%������������� �������������� ������������� 
clear all; close all;
%1.�������� ������ ��� ��������� ������� M ����������� �������
n=2; M=5;%����������� ������������ ������������ � ����� ������� 
%L - ���������� ����������� ����� � ������ ������ 
%dm - ��������, ������������ ������� ������� ����������� ����������� ������
%romin, romax -  ������� �������� ������������ ���������� ��� ������� ������ ����������
L=ones(1,M);%������ ����� ����������� ����� ����������� �������������� 
dm=3; romin=-0.9; romax=0.9; 
%����, �������������� ��������, ��������� � ������������ ���������� ����������� ������
ps=cell(1,M); mM=cell(1,M); D=cell(1,M); ro=cell(1,M);
for i=1:M, 
    ps{i}=ones(1,L(i))/L(i); D{i}=ones(1,L(i));  ro{i}=romin+(romax-romin)*rand(1,L(i)); 
end;
mM{1}=[0;0]; mM{2}=[0;dm]; mM{3}=[dm;0]; mM{4}=[dm;dm]; mM{5}=[-dm;-dm];      
Ni=50; NN=[Ni,Ni,Ni,Ni,Ni]; N=sum(NN);%������ ����������� ������ 
%2.������������ ��������� 
X=gen(n,M,NN,L,ps,mM,D,ro,0);
Nmi=0; Ns=zeros(1,M); XN=zeros(N,n);
for i=1:M, Nma=Nmi+NN(i); Ns(i)=Nma; XN(Nmi+1:Nma,:) =X{i}'; Nmi=Nma; end;
Qn = linkage(XN,'weighted','euclidean');%���������� ������������ 
figure(1); dendrogram(Qn);             %����������� ������������
idx = cluster(Qn,'maxclust',M); %���������� ������������� �� ������ ������������
%idx = clusterdata(XN,'maxclust',M,'linkage','ward'); %���������� ������������� �� ������ ������
%idx-������ �������������� ������ ������� ��������
%3.������ ������, ������������  �������� ������ � ��������� �������
[ercl,idxn,prM] = erclust(M,NN,idx);%������ ������
disp('������ �������� ������������� � �������� ������'); disp([prM,ercl]);

% ������� ���������� ����������� �����
% https://www.mathworks.com/help/stats/linkage.html#d117e536575
% + �������� ������������� � ������� ��������� (�� ��������)
Qn = linkage(XN,'weighted','squaredeuclidean');%���������� ������������ 
figure(1); dendrogram(Qn);             %����������� ������������
idx = cluster(Qn,'maxclust',M); %���������� ������������� �� ������ ������������
[ercl,idxn,prM] = erclust(M,NN,idx);%������ ������
disp('������ �������� ������������� � �������� ������'); disp([prM,ercl]);

% + �������� ������������� � ������� ���������
Qn = linkage(XN,'weighted','seuclidean');%���������� ������������ 
figure(1); dendrogram(Qn);             %����������� ������������
idx = cluster(Qn,'maxclust',M); %���������� ������������� �� ������ ������������
[ercl,idxn,prM] = erclust(M,NN,idx);%������ ������
disp('������ �������� ������������� � �������� ������'); disp([prM,ercl]);

% + �������� ������������� � ������� ���������
Qn = linkage(XN,'weighted','euclidean');%���������� ������������ 
figure(1); dendrogram(Qn);             %����������� ������������
idx = cluster(Qn,'maxclust',M); %���������� ������������� �� ������ ������������
[ercl,idxn,prM] = erclust(M,NN,idx);%������ ������
disp('������ �������� ������������� � �������� ������'); disp([prM,ercl]);

% + �������� ������������� � ������� ���������
Qn = linkage(XN,'weighted','seuclidean');%���������� ������������ 
figure(1); dendrogram(Qn);             %����������� ������������
idx = cluster(Qn,'maxclust',M); %���������� ������������� �� ������ ������������
[ercl,idxn,prM] = erclust(M,NN,idx);%������ ������
disp('������ �������� ������������� � �������� ������'); disp([prM,ercl]);

% + �������� ������������� � ������� ���������
Qn = linkage(XN,'weighted','mahalanobis');%���������� ������������ 
figure(1); dendrogram(Qn);             %����������� ������������
idx = cluster(Qn,'maxclust',M); %���������� ������������� �� ������ ������������
[ercl,idxn,prM] = erclust(M,NN,idx);%������ ������
disp('������ �������� ������������� � �������� ������'); disp([prM,ercl]);

% + �������� ������������� � ������� ���������
Qn = linkage(XN,'weighted','cityblock');%���������� ������������ 
figure(1); dendrogram(Qn);             %����������� ������������
idx = cluster(Qn,'maxclust',M); %���������� ������������� �� ������ ������������
[ercl,idxn,prM] = erclust(M,NN,idx);%������ ������
disp('������ �������� ������������� � �������� ������'); disp([prM,ercl]);

% + �������� ������������� � ������� ���������
Qn = linkage(XN,'weighted','minkowski');%���������� ������������ 
figure(1); dendrogram(Qn);             %����������� ������������
idx = cluster(Qn,'maxclust',M); %���������� ������������� �� ������ ������������
[ercl,idxn,prM] = erclust(M,NN,idx);%������ ������
disp('������ �������� ������������� � �������� ������'); disp([prM,ercl]);

% + �������� ������������� � ������� ���������
Qn = linkage(XN,'weighted','chebychev');%���������� ������������ 
figure(1); dendrogram(Qn);             %����������� ������������
idx = cluster(Qn,'maxclust',M); %���������� ������������� �� ������ ������������
[ercl,idxn,prM] = erclust(M,NN,idx);%������ ������
disp('������ �������� ������������� � �������� ������'); disp([prM,ercl]);

% ������� ���� ������ �������� ������������� �������� 1 � �������� ������
% ���� �����������

figure(2); grid on; hold on;
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
% legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4',1); hold off;
