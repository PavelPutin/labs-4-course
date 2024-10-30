%���� pr33_est_Parzen.������ ��������� ������������� ������������ ������� �������
clear all; close all;

%% ���������� ������

% 1. �������� ������
n=1; %n-����������� ������� ����������
N=1000; %���������� ������������ ��� ������ ��������
r=0.5;
h_N=N^(-r/n); %������ ��������� ������� ����
kl_kernel=12;%���� ������ ���� ������ (��. �������� ������� vkernel)

% 2. ��������� ��������� ������� � �������� ��������� ��������� ��� ����������� ������
x0=-3:0.05:3; %������� ����� �������� (������� �������� ��, ��� ������� ��������������� ������)
p=zeros(1,length(x0)); 

% ������������� ���� �� ����������

% % 2.�) ����������� ������������� �� � �����������: m=0, D=1;
% XN=randn(1,N);
% p=exp(-x0.^2/2)/sqrt(2*pi);%��� ����������� ��������� (�����������)

% % 2.�) ����������� ������������� �� � ����������� a=0, b=1;
% XN=rand(1,N); 
% ind1=logical(x0>0 & x0<=1);
% p(ind1)=1;%��� ����������� ��������� (�����������)

% 2.�) ������������� ������������� �� � ���������� b=1;
XN=-log(rand(1,N));
ind1=logical(x0>0);
p(ind1)=exp(-x0(ind1));%��� ����������� ���������  (�������������)

% 3. ������ ��������� �� �������
p_=vkernel(x0,XN,h_N,kl_kernel);%������ ���������

% 4. ����������� �������� ��������� ������������� � �� ������
figure; grid on; hold on;
axis([min(x0), max(x0), 0, max(p)+0.1]);%��������� ������ ���� ������� �� ����
plot(x0,p,'-b',x0,p_,'-+k');
title('��������� ������������� � �� ������','FontName','Courier','FontSize',14);
xlabel('x','FontName','Courier'); ylabel('p','FontName','Courier');
strv1='N= '; strv2=num2str(N); strv3='  h= '; strv4=num2str(h_N);
switch kl_kernel, case 11, strv5=' gauss'; case 12, strv5=' gauss'; case 2, strv5=' exp';...
    case 3, strv5=' rect';  case 4, strv5=' triang'; end;
text(-2.5, 0.97*max(p), [strv1, strv2, strv3, strv4,strv5], 'HorizontalAlignment','left',...
      'BackgroundColor',[.8 .8 .8],'FontSize',12);
legend('p ','p~'); hold off;

%% ��������� ������

% 1. �������� ������
n=2; %n-����������� ������� ����������
N=10000; %���������� ������������ ��� ������ ��������
r=0.5;
h_N=N^(-r/n); %������ ��������� ������� ����
kl_kernel=12;%���� ������ ���� ������ (��. �������� ������� vkernel)

% 2.��������� �������� ��������� ��������� (� ���� ����� ��������) ��� ���������� ������
% ��������� ������������� ����� ����������� ��������� ��������;
M=3; %���������� ����������� � �����
ps=[0.2,0.2,0.6]; %����������� ��������� �� ��������� ����� � �����
% ������ ������� ���������� ��� �����
D=0.2; ro=-log(0.7); %��������� � ����������� ���������� c������� ��������� 
% ������������ �������������� �������� ����������� �����
m1=[0;0]; m2=[1;0]; m3=[0;1]; m=[m1,m2,m3];
% �������������� ������� ����������� �����
for i=1:n, for j=1:n,
        C(i,j)=D*exp(-ro*abs(i-j));
end;end;
x1=[-2:0.1:3]; x2= [-2:0.1:3]; %������� �������� ��, ��� ������� ��������������� ������
[X1,X2]=meshgrid(x1,x2); x=[X1(:) X2(:)]'; % ������� � � Y ��������� ��������
% �������� ��������� ���������
p=ps(1)*mvnpdf(x',m1',C)+ps(2)*mvnpdf(x',m2',C)+ps(3)*mvnpdf(x',m3',C);
pi=reshape(p,length(x1),length(x2));%������� �������� ��������� �������������

% 3. ����������� ������� ��������� �������������
figure; grid on; hold on;
caxis([min(pi(:))-0.5*range(pi(:)),max(pi(:))+0.5*range(pi(:))]);%��������� ����� 
axis([min(x1) max(x1)  min(x1) max(x1) 0 max(pi(:))+0.1]);%������� ���� ������� 
surf(x1,x2,pi);
title('������������� ����� ����������� ��������','FontName','Courier','FontSize',14);
xlabel('x1','FontName','Courier'); ylabel('x2','FontName','Courier'); zlabel('pi');
strv1='N= '; strv2=num2str(N);strv3=' h= '; strv4=num2str(h_N);
switch kl_kernel, case 11, strv5=' gauss i'; case 12, strv5=' gauss c'; end;
text(1, -0.5, 0.5*max(pi(:)),[strv1,strv2,strv3,strv4,strv5], 'HorizontalAlignment','left',...
    'BackgroundColor',[.8 .8 .8],'FontSize',12);
hold off;

% 4. ��������� �������
XN=zeros(n,N);
for i=1:N,%��������� ��������� ������� 
     u=rand;
     %������ �������������� � ���������� �����
     if u<ps(1), t=1; elseif u<ps(1)+ps(2), t=2; else t=3; end;
     XN(:,i)=randncor(n,1,C)+m(:,t);
end;

% 5. ������ ��������� �� �������
p_=vkernel(x,XN,h_N,kl_kernel);%������ ���������
%������� ��������  ������ ��������� �������������
pv=reshape(p_,length(x1),length(x2));

% 6. ����������� ������� ������ ��������� �������������
figure; grid on; hold on;
caxis([min(pv(:))-0.5*range(pv(:)),max(pv(:))+0.5*range(pv(:))]);%��������� �������� �����
axis([min(x1),max(x1),min(x1),max(x1),0,max(pi(:))+0.1]);%��������� ������ ���� ������� �� ����
surf(x1,x2,pv);
title(' ������ ��������� ������������� ����� ','FontName','Courier','FontSize',14);
xlabel('x1','FontName','Courier'); ylabel('x2','FontName','Courier'); zlabel('p~');
text(1, -0.5, 0.5*max(pv(:)),[strv1,strv2,strv3,strv4,strv5], 'HorizontalAlignment','left',...
    'BackgroundColor',[.8 .8 .8],'FontSize',12);
hold off;
% 7. ������������ ����� ����������� ������ ��������� ������������� � �� ������
figure;hold on;
[Cv,h]=contour(x1,x2,pi,[0.001, 0.01, 0.5*max(pv(:))]);
clabel(Cv,h);
title('����� ����������� ������ ���������','FontName','Courier','FontSize',14);
xlabel('x1','FontName','Courier'); ylabel('x2','FontName','Courier'); 
text(0, 0, [strv1,strv2,strv3,strv4,strv5], 'HorizontalAlignment','left',...
    'BackgroundColor',[.8 .8 .8],'FontSize',12);
hold off;
figure;hold on;
[Cv,h]=contour(x1,x2,pv,[0.001, 0.01, 0.5*max(pv(:))]);
clabel(Cv,h);
title('����� ����������� ������ ������ ���������','FontName','Courier','FontSize',14);
xlabel('x1','FontName','Courier'); ylabel('x2','FontName','Courier'); 
strv1='N= '; strv2=num2str(N);strv3=' h= '; strv4=num2str(h_N);
text(0, 0, [strv1,strv2,strv3,strv4,strv5], 'HorizontalAlignment','left',...
    'BackgroundColor',[.8 .8 .8],'FontSize',12);
hold off;
