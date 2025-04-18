clear all; close all;
%%% ������ ���.1. ��������� ���������� ������ ���������� ��������� 
% ������������� ��������� �������� ��� ������������� ������ �������.
% ��������� ������ ����������� ������ ���������� �� �������� ��������� ������� ������� 

%% ����� ������ ���������� ������

% ����� �������� ������������ �������� �������� r �� ������ �������
% ����������� �������� ������� �������
RR = 0.1 : 0.1 : 0.9;
err = RR * 0;   % ������ �������� ������ ����������� ������

% ����� ����������� ���� �� ����� ��������� RR
for tt = 1 : numel(RR)
% 1. �������� ������
n=1; %n-����������� ������� ����������
N=1000; %���������� ������������ ��� ������ ��������
r=RR(tt);    % ����� ����������� ��������� �������� �� ������� RR
h_N=N^(-r/n); %������ ��������� ������� ����
% ����� ���� ������� ������� (����) � ����������� �� ����� �������
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
% ����� ��������� ���������� ������
err(tt) = mean(abs(p(:) - p_(:)));
end
% ����� ������ �.4. ������� ����������� ������ �� �������� r
figure;
plot(RR, err);  % �� �������� �� �����������, ��� ����������� ������� - � ���� ��������� �������� r
