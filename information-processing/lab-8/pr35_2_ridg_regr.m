%���� pr35_2_ridg_regr. ���������� ��������� ���������
clear all; close all;
%1.������� �������� ������
N=100;%����� ��������� ������� 
n=2;%����������� ������� ������� ����������
K=10;%����� ����� ����� ��������� ��������� �������������
eps=0.0000001;%�������� ������� ������������������ ������� ����������
D=0.1; %��������� ������ ��������� �������� ���������� 
delt=3*sqrt(D*N); %��������� ������� �������
C=eye(n);%������� ���������� �������� �������
C(n,:)=(1-eps)*ones(1,n); C(:,n)=(1-eps)*ones(1,n); C(n,n)=n-1;
xmin=-3; xmax=3; %������� ������� ����������� ������� ����������
A=[-1,(1:n)]';%������ ����������� ���������� ���������
%2.��������� ��������� ������� ������ � ������� ����� ������
[XN,m]=randncor(n,N,C);
X=[ones(N,1),XN']; YN=X*A+sqrt(D)*randn(N,1);
XtX=X'*X;
betta1=max(eig(XtX))/1.0e+03;%������������� �������� ��������� �������������
rc=cond(C); rx=cond(XtX); rxb=cond(XtX+betta1*eye(n+1));
disp(['����� ��������������� ������: �, XtX,XtX+bI']);
disp([rc,rx,rxb]);

%��������� ����� ������
XNn=xmin+(xmax-xmin)*rand(n,N); 
Xn=[ones(N,1),XNn']; YNn=Xn*A+sqrt(D)*randn(N,1);
Mb=zeros(K,1); Rb=zeros(K,1); Rbn=zeros(K,1); Ea=zeros(K,1);fm=zeros(1,K);
betta=0.0000001;%��������� �������� ��������� �������������
%3.�������������� ��������� ��� ������ ��������� ��������� �������������
for k=1:K, %���� �� ��������� ��������� �������������
    betta=betta*10; Mb(k)=log10(betta);
    %����������� ������������� ��������� ���������
    a_=(X'*X+betta*eye(n+1))\(X'*YN);
    %������ ������� � ��� ���������� ������������� ������
    Rb(k)=sqrt((X*a_-YN)'*(X*a_-YN));
    Rbn(k)=sqrt((Xn*a_-YNn)'*(Xn*a_-YNn));
    Ea(k)=sqrt((A-a_)'*(A-a_));
end;%���� �� k
%4.������������ �����������
%������������ ������� ������ � 3D
if n==3,
     figure(1);set(gca,'FontSize',12); grid on; hold on;
     pg3=plot3(XN(1,:),XN(2,:),XN(3,:),'ro'); set(pg3,'LineWidth',1.25);
     xlabel('x1','FontName','Courier');ylabel('x2','FontName','Courier');zlabel('x3','FontName','Courier');
     title('���������������� ����������� ������� ������','FontName','Courier'); hold off;   
 end;
%���������� �������� ��� ������ ��������� ������������� 
figure(2);set(gca,'FontSize',12); grid on; hold on;
%ms=max([max(Rb),max(Rbn),max(Ea),delt])+1;
ms=3*delt;
axis([min(Mb) max(Mb) 0 ms]);
pg=plot(Mb,Rb,'-b^',Mb,Rbn,'--b^',Mb,Ea,'--r+',Mb,delt*ones(1,K),'--k');
set(pg,'LineWidth',1.25);
bl=log10(betta1);%������������� �������� betta �� ��� OX
line('Xdata',[bl,bl],'Ydata',[0,ms],'LineWidth',1.5);
title('������������� ������ ���������� �������� ','FontName','Courier');
xlabel('�������� ��������� ������������� (betta)','FontName','Courier')
str1='N='; str2=num2str(N);str3=' eps='; str4=num2str(eps);str5=' D='; str6=num2str(D);
text(Mb(1)+1,ms-delt,[str1, str2, str3, str4, str5, str6], 'HorizontalAlignment','left','BackgroundColor',[.8 .8 .8],'FontSize',12);
legend('Rb','Rbn','Ea','delt','bl');hold off;   
