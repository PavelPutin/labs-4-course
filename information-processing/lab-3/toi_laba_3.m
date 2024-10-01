%% ���� pr53_rec_gaus_uneq. ������ � ������ ���������� ������������� ��� � 
% ���������� ��������� ����������
clear all;
close all;

%%  1. ������� �������� ������
n = 2;
M = 2; % ����������� ������������ ������������ � ����� �������
K = 1000 ; % ���������� �������������� ���������
% ��������� �����������, �������������� �������� � ������� ���������� �������
dm = 2.0; % ���������� ����� ��������������� ���������� ������� �� ������������ ����
C = zeros(n, n, M);
C_ = C; % ������� ���������� ������� ��������� ��������� �������
pw = [0.4 0.6];
pw = pw / sum(pw);
D = 3 * eye(2);
m = [2 1; -1 -2; -2 1]';
C(:,:,1) = [3 -1; -1 3];
C(:,:,2) = [5 3; 3 5];
for k = 1 : M
    C_(:,:,k) = C(:, :, k) ^ -1; 
end
np = sum(pw);
pw = pw / np; % ���������� ������������� ������� ��������� ������������

%%  2. ������ ������ ������������ ������ �������������
PIJ = zeros(M);
PIJB = zeros(M);
mg = zeros(M);
Dg = zeros(M);
l0_ = zeros(M);
for i = 1 : M
    for j = i + 1 : M
           dmij = m(:, i) - m(:, j); 
           l0_(i,j) = log(pw(j) / pw(i)); 
           dti = det(C(:, :, i));
           dtj = det(C(:, :, j));
           trij = trace(C_(:, :, j) * C(:, :, i) - eye(n));
           trji = trace(eye(n) - C_(:, :, i) * C(:, :, j));
           trij_2 = trace((C_(:, :, j) * C(:, :, i) - eye(n)) ^ 2);
           trji_2 = trace((eye(n) - C_(:, :, i) * C(:, :, j)) ^ 2);
           mg1 = 0.5 * (trij + dmij' * C_(:, :, i) * dmij - log(dti / dtj)); 
           Dg1 = 0.5 * trij_2 + dmij' * C_(:, :, j) * C(:, :, i) * C_(:, :, j) * dmij; 
           mg2 = 0.5 * (trji - dmij' * C_(:, :, j) * dmij + log(dtj / dti)); 
           Dg2 = 0.5 * trji_2 + dmij' * C_(:, :, i) * C(:, :, j) * C_(:, :, i) * dmij; 
           sD1 = sqrt(Dg1);
           sD2 = sqrt(Dg2);
           PIJ(i, j) = normcdf(l0_(i, j), mg1, sD1);
           PIJ(j, i) = 1 - normcdf(l0_(i, j), mg2, sD2);
           mu2 = (1 / 8) * dmij' * ((C(:, :, i) / 2 + C(:, :, j) / 2) ^ -1) * dmij...
               + 0.5 * log((dti + dtj) / ( 2 * sqrt(dti * dtj))); % ���������� ����������
           PIJB(i, j) = sqrt(pw(j) / pw(i)) * exp(-mu2);
           PIJB(j, i) = sqrt(pw(i) / pw(j)) * exp(-mu2); % ������� �������
    end
    PIJB(i, i) = 1 - sum(PIJB(i, :));
    PIJ(i, i) = 1 - sum(PIJ(i, :)); % ������ ������� ����������� ����������� �������������
end
%% 3. ������������ ��������� ������� �������������� ���������
x = ones(n, 1);
u = zeros(M, 1);
Pc_ = zeros(M); % ����������������� ������� ������������ ������
for k = 1 : K % ���� �� ����� ���������
    for i = 1 : M % ���� �� �������
        [x, px] = randncor(n, 1, C(:, :, i));
        x = x + m(:, i); % ��������� ������ i-�� ������
        for j = 1 : M % ���������� �������� ����������� �������
            u(j) = -0.5 * (x - m(:, j))' * C_(:, :, j) * (x - m(:, j)) - 0.5 * log(det(C(:, :, j))) + log(pw(j));
        end
        [ui, iai] = max(u); % ����������� ���������
        Pc_(i, iai) = Pc_(i, iai) + 1; % �������� ���������� �������������
    end
end
Pc_ = Pc_ / K;
disp('������������� ������� ������������ ������');
disp(PIJ);
disp('������� ������������ ������ �� ������ ������� �������');
disp(PIJB);
disp('����������������� ������� ������������ ������');
disp(Pc_);
%% 4. ������������ �������� �������� ������� ��� ���������� ������
if n == 2
      Es1 = pw(1) * PIJ(1, 2) + pw(2) * PIJ(2,1);
      Es2 = sqrt(pw(1) * pw(2)) * exp(-mu2); % ������� ������� ��� ��������� ������
      Es3 = pw(1) * Pc_(1, 2) + pw(2) * Pc_(2, 1);
      disp('������ ��������� ������');
      disp([Es1, Es2, Es3]); % ����������� ������ ��������� ������
      xmin1 = -3 * sqrt(max(D(1, :))) + min(m(1, :));
      xmax1 = 3 * sqrt(max(D(1, :))) + max(m(1, :));
      xmin2 = -3* sqrt(max(D(2, :))) + min(m(2, :));
      xmax2 = 3* sqrt(max(D(2, :))) + max(m(2, :));
      x1 = xmin1 : 0.1 : xmax1;
      x2 = xmin2 : 0.1 : xmax2;
      axis([xmin1, xmax1, xmin2, xmax2]); % ��������� ������ ���� ������� �� ����
      figure(1);
      hold on;
      grid on;
      [X1, X2] = meshgrid(x1, x2); % ������� �������� ��������� ���������� �������
      x12 = [X1(:), X2(:)]; 
      for i=1:M
          f2 = mvnpdf(x12, m(:, i)', C(:, :, i)); % ������ �������� ��������� �������������
          f3 = reshape(f2, length(x2), length(x1)); % ������� �������� ��������� �������������
          [Ch, h] = contour(x1, x2, f3, [0.01, 0.5 * max(f3(:))], 'Color', 'b', 'LineWidth', 0.75);
          clabel(Ch, h);
          for j = i + 1 : M % ����������� ����������� ������
              wij = C_(:, :, i) * m(:, i) - C_(:, :, j) * m(:, j);
              wij0 = -0.5 * (m(:, i)' * C_(:, :, i) * m(:, i) - m(:, j)' * C_(:, :, j) * m(:, j));
              f4= wij' * x12' + wij0 - 0.5 * log(det(C(:, :, i)) / det(C(:, :, j)));
              fd = -0.5 * (C_(:, :, i) - C_(:, :, j)) * x12';
              f4 = f4 + sum(x12' .* fd);
              f5 = reshape(f4, length(x2), length(x1));
              [Ch_, h_] = contour(x1, x2, f5, l0_(i, j), 'Color', 'k', 'LineWidth', 1.25); 
          end
      end
    set(gca, 'FontSize', 13);
    title('������� ����������� ������� � ����������� �������', 'FontName', 'Courier');
    xlabel('x1', 'FontName', 'Courier');
    ylabel('x2', 'FontName', 'Courier'); 
    strv1 = ' pw=';
    strv2 = num2str(pw, '% G');
    text(0, 0, [strv1, strv2], 'HorizontalAlignment', 'left', 'BackgroundColor', ...
        [.8 .8 .8], 'FontSize', 12);
    legend('wi', 'gij(x)=0');
    hold off;
end







