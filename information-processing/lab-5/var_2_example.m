%���� pr33_est_Parzen.������ ��������� ������������� ������������ ������� �������
clear all; close all;

%% ����� ������ ���������� ������
% ����� �������� ���� ������� ������� � ����� vkernel.m (������ vkernel_var2)

% 1. �������� ������
n=1; %n-����������� ������� ����������
N=1000; %���������� ������������ ��� ������ ��������
r=0.5;
h_N=N^(-r/n); %������ ��������� ������� ����
% ���� 5 - ���.2.�), 6 - ���.2.�)
kl_kernel=6;%���� ������ ���� ������ (��. �������� ������� vkernel_var2)

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
p_=vkernel_var2(x0,XN,h_N,kl_kernel);%������ ���������

% ����� ����� �������� ���������� ������������������ ������ ����������
err = sqrt(mean((p - p_) .^ 2))
% ����� ��������� �������� ��������� ������ ��� ���� ����� ������� �������
% (��� ������ kl_kernel) � ������ ����� �� ��� ����� (� ���������� �������)

% 4. ����������� �������� ��������� ������������� � �� ������
figure; grid on; hold on;
axis([min(x0), max(x0), 0, max(p)+0.1]);%��������� ������ ���� ������� �� ����
plot(x0,p,'-b',x0,p_,'-+k');
title('��������� ������������� � �� ������','FontName','Courier','FontSize',14);
xlabel('x','FontName','Courier'); ylabel('p','FontName','Courier');
strv1='N= '; strv2=num2str(N); strv3='  h= '; strv4=num2str(h_N);
% ����� ����� �������� ����� ������
% switch kl_kernel, case 11, strv5=' gauss'; case 12, strv5=' gauss'; case 2, strv5=' exp';...
%     case 3, strv5=' rect';  case 4, strv5=' triang'; end;
% text(-2.5, 0.97*max(p), [strv1, strv2, strv3, strv4,strv5], 'HorizontalAlignment','left',...
%       'BackgroundColor',[.8 .8 .8],'FontSize',12);
% legend('p ','p~'); hold off;