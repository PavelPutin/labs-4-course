%���� pr35_1_lin_regr. ������������� ����������� ������� MatLab ��� ���������� 
%�������� ���������
clear all; %close all;
%1.������� �������� ������
var1=2;%��� ����������������� ������� (1-�������; 2-������������� ���)%var1 
ng=5;% ������� �������� (����), ng+1 - ���������� ������������� �������� (����)
N=100;%����� ��������� ������� 
n=1;%����������� ������� ������� ���������� (� ������ ������� �����������)
D=1; %��������� ������ ��������� �������� ����������
gamma=0.01;%������� ���������� ��� �������� ������� �� �������� ������
xmin=0; xmax=1; dx=0.01;%������� � ������������ ������� ����������� ������� 
xd=xmin:dx:xmax; ld=length(xd);
%2.��������� ��������� ������� ������
XN=xmin+(xmax-xmin)*rand(N,n); 
YN=zeros(N,1); a=zeros(ng+1,1);
if var1==1,
     %�������� ������ ��� ��������
     a(1)=1; a(2)=5;
     a(3)=10; a(4)=0.5*10^3; a(5)=0.05*10^4; a(6)=-0.01*10^5; 
end;
if var1==2,
     %�������� ������ ��� �������������� ����
     a(1)=0.5; a(2)=1;
     a(3)=10; a(4)=-5; a(5)=0.5; a(6)=-1; 
end;
%������� ���������������� ������� 
if var1==1,
    p=fliplr(a'); %� �������� �������
    YN=polyval(p,XN)+sqrt(D)*randn(N,1);
end;
if var1==2,
    YN=a(1)*ones(N,1)+a(2)*sin(pi*XN)+a(3)*sin(2*pi*XN)+a(4)*sin(3*pi*XN)...
     +a(5)*sin(4*pi*XN)+a(6)*sin(5*pi*XN)+sqrt(D)*randn(N,1);
end;
%3.��������� � ������� ���������� ������������� ���������
X=[ones(N,1),XN];
[a_,aint,r,rint,stat]=regress(YN,X,gamma);
%a_-������ ������������� ��������� (n+1)x1;
%bint-������� ������������ ������ ������������� (n+1)x2;%r-������ �������� Nx1;
%rint-������� ������������� ���������� �������� Nx2;
%stat- ������, ���������� ��������: ��������� R^2 � F, p-��������, RSS/(N-n-1)
RSS=sum(r.^2);
mY=mean(X*a_); 
ESS=sum((X*a_-mY).^2);
R2=ESS/(RSS+ESS);%����������� ������������
F=(ESS/n)/(RSS/(N-n-1));%���������� ������
fgamma=finv(1-gamma,n,N-n-1);%�������� F-����������, ����������� � ������������ gamma
p_value=1-fcdf(F,n,N-n-1);%������ ����������� ���������� ����������� �������� F

%4.����� ���������� ���������
XtX=X'*X;
betta=max(eig(XtX))/100%������������� �������� ��������� �������������
%����������� ������������� ��������� ���������
a_lms=(X'*X+betta*eye(n+1))\(X'*YN);
%������ ������� � ��� ���������� ������������� ������
Rb=(X*a_lms-YN)'*(X*a_lms-YN);

disp('��������� ���������� (����������� ������) ������������� �������� ���������:')
disp(RSS);
disp('��������� ���������� (����������� ������) ������ ���������� ���������:')
disp(Rb);

%5.������������ �����������
%������������ ������� ������ � 3D
if n==3,
     figure(1);set(gca,'FontSize',12); grid on; hold on;
     pg3=plot3(XN(1,:),XN(2,:),XN(3,:),'ro'); set(pg3,'LineWidth',1.25);
     xlabel('x1','FontName','Courier');ylabel('x2','FontName','Courier');zlabel('x3','FontName','Courier');
     title('���������������� ����������� ������� ������','FontName','Courier'); hold off;   
 end;

%�������� ����������� ��������� � �����
% disp(['����������� ����������: R^2, F, p_value, RSS/(N-n-1)']);
% disp([R2,F,p_value,RSS/(N-n-1)]);disp(stat);
%4.���������� ��������� � ������ ����������������� �������
x=[ones(ld,1),xd']; y=x*a_; y_lms=x*a_lms;
ymi=x*aint(:,1); yma=x*aint(:,2);
ymin=min(ymi); ymax=max(yma);
%5.������������ �����������
figure;set(gca,'FontSize',12);grid on; hold on;
axis([xmin xmax ymin-0.1 ymax+0.1]);%��������� ������ ���� ������� �� ����
pg=plot(XN,YN,'ro',xd,y,'-b',xd,y_lms,'-g',xd,ymi,'--k',xd,yma,'--k');
set(pg,'LineWidth',1.25);
title('���������� ������������� �����������','FontName','Courier');
xlabel('X','FontName','Courier');ylabel('Y','FontName','Courier');
str1='N='; str2=num2str(N);
str3=' ng='; str4=num2str(ng);
str5=' D='; str6=num2str(D);
str7=' gamma='; str8=num2str(gamma);
str9=' p-val='; str10=num2str(p_value);
text(xmin+0.1, 0.5*ymax,[str1, str2, str3, str4, str5, str6,str7,str8,str9,str10], 'HorizontalAlignment',...
    'left','BackgroundColor',[.8 .8 .8],'FontSize',12);
legend('XN-YN','y=f(x)','y=f_{lms}(x)','y-dy','y+dy');hold off;




