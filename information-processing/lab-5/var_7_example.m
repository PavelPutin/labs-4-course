clear all; close all;
%%% ������ ���.7. ����������� ������ ��������� ������������� ���������� ���������� �������,
% ��������� �������� �������� �� ������ ���� ��������� �������. ���������� �����������
% �� �������� ������������������ ������ ���������� �������� ��������� k.

%% ����� ������ ��������� ������

% ����� �������� ������������ �������� �������� g �� ������ �������
% ����������� �������� K
GG = 0.1 : 0.1 : 0.9;
err = GG * 0;   % ������ �������� ������ ����������� ������

% ����� ����������� ���� �� ����� ��������� RR
for tt = 1 : numel(GG)
    % 1. �������� ������
    n=2; %n-����������� ������� ����������
    N=1000; %���������� ������������ ��� ����� � ��������
    gm=GG(tt); % ����� ����������� ��������� �������� �� ������� GG
    k=round(N^gm); %k - ����� ��������� �������

    % 2.��������� �������� ��������� ��������� (� ���� ����� ��������) ��� ���������� ������
    % ��������� ������������� ����� ����������� ��������� ��������;
    M=5; %���������� ����������� � �����
    ps=[0.1,0.2,0.3,0.1,0.1]; %����������� ��������� �� ��������� ����� � �����
    % ������ ������� ���������� ��� �����
    D=0.2; ro=-log(0.7); %��������� � ����������� ���������� c������� ��������� 
    % ������������ �������������� �������� ����������� �����
    m1=[0;0]; m2=[1;0]; m3=[0;1]; m4=[-1;0]; m5=[-1;-1]; m=[m1,m2,m3,m4,m5];
    % �������������� ������� ����������� �����
    for i=1:n, for j=1:n,
            C(i,j)=D*exp(-ro*abs(i-j));
    end;end;
    x1=[-2:0.1:3]; x2= [-2:0.1:3]; %������� �������� ��, ��� ������� ��������������� ������
    [X1,X2]=meshgrid(x1,x2); x=[X1(:) X2(:)]'; % ������� � � Y ��������� ��������
    % �������� ��������� ���������
    p=ps(1)*mvnpdf(x',m1',C)+ps(2)*mvnpdf(x',m2',C)+ps(3)*mvnpdf(x',m3',C)+ps(4)*mvnpdf(x',m4',C)+ps(5)*mvnpdf(x',m5',C);
%     pi=reshape(p,length(x1),length(x2));%������� �������� ��������� �������������

    % 3. ������ ����������� ������� ��������� �������������
    
    % 4. ��������� �������
    XN=zeros(n,N);
    for i=1:N,%��������� ��������� ������� 
         u=rand;
         %������ �������������� � ���������� �����
         if u<ps(1), t=1; elseif u<ps(1)+ps(2), t=2; elseif u<ps(1)+ps(2)+ps(3), t=3; elseif u<ps(1)+ps(2)+ps(3)+ps(4), t=4; else t=5; end;
         XN(:,i)=randncor(n,1,C)+m(:,t);
    end;

    % 5. ������ ��������� �� ������ � ��������� �������
    p_=vknn(x,XN,k);%������ ���������
    %������� �������� ��������� ���������� �������
    % pv=reshape(p_,length(x1),length(x2));%������� �������� ������ ��������� �������������
    
    % ����� ��������� ������������������ ������
    err(tt) = sqrt(mean((p(:) - p_(:)).^2));
end
% ����� ������ �.6,7. ������� ����������� ������ �� �������� g
figure;
plot(GG, err);  % �� �������� �� �����������, ��� ����������� ������� - � ���� ��������� �������� g