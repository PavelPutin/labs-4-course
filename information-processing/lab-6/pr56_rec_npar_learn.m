% ���� pr56_rec_npar_learn. �������� ���������� ������������� �� ������ ������
% ���������� ������������� �������� ������� � k - ��������� �������
clear all;
close all;
%% 1. ������� �������� ������
n = 2;M = 2; % ����������� ������������ ������������ � ����� �������
H = 1; % ���������� �������������� ��������� �������� ��������
K = 1000; % ���������� �������������� ��������� ��������� �������������
pw = ones(1, M) / M; % ��������� ����������� ������� (�������� �� ���������)
% �������� ������ ��� ��������� ��������� ������� 
% dm - ��������, ������������ ������� ������������������ ����������� ������
% DM - ��������, ������������ ����� �������� ����������� ������� �� ����
% L - ���������� ����������� � ����� ������� ������ 
L = [2, 2];
dm = 2;
DM = 1;
ps = {[0.5; 0.5],[0.5; 0.5]};
mM = {[2 2], [1 -1]};
C = [5 1; 1 5];
kl_kernel = 4;

switch kl_kernel 
    case 11 
        disp("����������� ������� c �������������� ������������ �������"); 
    case 12 
        disp("����������� ������� c �������������� ������� ����������"); 
    case 2 
        disp("������������� �������"); 
    case 3 
        disp("������� ������������� �������"); 
    case 4 
        disp("������� ����������� �������"); 
end

r = 0.25;
gm = 0.25; % ��������� ������ (��.�������� ������� vkernel, vknn)
%% 2. ��������� ��������� ������ � ����� � ���������� ������� �������
Nn = [10, 20, 30, 40, 50, 100, 150, 200, 250]; 
ln = length(Nn);
Esth1 = zeros(1, ln); % ��������� ������ ������� ������� ����������� ��������
for nn = 1:ln % ���� � ���������� ������ ��������� �������
    NN = Nn(nn) * ones(1, M);
    N = Nn(nn);
    h_N = N^(-r / n); % ������� ���� �������
    k = 2 * round(N^gm) + 1; % k - ����� ��������� �������
    for h = 1:H % ���� �������������� ��������� �������� �������� 
        XN = gen(n, M, NN, L, ps, mM, C, 0); % ��������� ��������� �������
        
        % 2.1. ����������� ������������ ������ ������� ����������� ��������
        Pc1 = zeros(M);
        p1_ = zeros(M, 1);
        for i = 1:M % ���������� ������ ����������� ��������
             XNi = XN{i};
             XNi_ = zeros(n, N - 1);
             indi = [1:i - 1, i + 1:M];
             for j = 1:N
                x = XNi(:, j);
                indj = [1:j - 1, j + 1:N]; % ������� ��������� ������ i - �� ������
                XNi_(:, 1:j - 1) = XNi(:, 1:j - 1);
                XNi_(:, j:end) = XNi(:, j + 1:end);
                p1_(i) = vkernel(x, XNi_, h_N, kl_kernel); % ������ �������
                for t = 1:M - 1
                     ij = indi(t);
                     p1_(ij) = vkernel(x, XN{ij}, h_N, kl_kernel);
                end
                [ui1, iai1] = max(p1_);
                Pc1(i, iai1) = Pc1(i, iai1) + 1;
             end
             Pc1(i, :) = Pc1(i, :) / N;
        end
        disp("������� ������ c ������� ������� " + num2str(N));
        disp(Pc1);
        Esth1(nn) = Esth1(nn) + (1 - sum(pw.* diag(Pc1)')); % ��������� ������
    end % �� h
end % �� nn
Esth1 = Esth1 / H;
%% 4. ������������ ������������ ������������ ������
% figure;
% grid on;
% hold on;
% ms = max(Esth1);
% axis([Nn(1), Nn(ln), 0, ms + 0.001]); % ��������� ������ ���� ������� �� ����
% p = plot(Nn, Esth1, '-b');
% set(p, 'LineWidth', 1.0);
% title('��������� ����������� ������ �� ������ ������ ������� � �������������� ������ ����������� ��������', 'FontName', 'Courier', 'FontSize', 14);
% xlabel('N', 'FontName', 'Courier', 'FontSize', 14);
% ylabel('Es', 'FontName', 'Courier', 'FontSize', 14);
% hold off;