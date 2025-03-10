% ���� pr54_rec_bin. ������ � ������ ���������� ������������� ������� �� ���������� 
% ��������� (�� ������� ������������� �������� �����������)

% ������ ��� ��������� ������� ����������� �������

clear all;
close all;
%% 1. ������� �������� ������
n = 35; % ���������� ��������� (������ �� ������� �����������)
M = 2;
s = zeros(n, M); % ���������� ������� � ��������� ��������
% 1.1. ������� �������� �������
letterP =  [1 1 1 1 0 ...
            1 0 0 0 1 ...
            1 0 0 0 1 ...
            1 1 1 1 0 ...
            1 0 0 0 0 ...
            1 0 0 0 0 ...
            1 0 0 0 0 ]';
letterM =  [1 0 0 0 1 ...
            1 1 0 1 1 ...
            1 0 1 0 1 ...
            1 0 0 0 1 ...
            1 0 0 0 1 ...
            1 0 0 0 1 ...
            1 0 0 0 1 ]';
% ����� P � ����� M
s(:, 1) = letterP; 
s(:, 2) = letterM;
% 1.2. ������� ���������� ������������
pw = [0.7, 0.3]; % ��������� ����������� �������
np = sum(pw);
pw = pw / np; % ���������� ������������� ������� ��������� ������������
N = 20; % ���������� ����� ��������� ������������ ��������� - pi
K = 1000; % ���������� ����������
pi = zeros(1, N); % ������ ������������ ��������� �������
% 1.3. ������� ����. � ����. ������ ������� � ������� ����
p12th = pi; 
p21th = pi;
p12ex = pi;
p21ex = pi;
% 1.4. ������� ������
Pc_ = zeros(M, M, N); % ����������������� ������� ������������ ������

% ���� �� ��������� ����������� ��������� ��������� ��������
for t = 1 : N
    pi(t) = (1 / N) * (t - 1);
    pI = pi(t); % ����������� ��������� �������� (�������)
    
	% 2.������ ��������� ������� � ������ ������������� ������������ ������
    if pI == 0
        pI = 0.0001;
    end % ������������� ����������� �������
    if pI == 0.5
        pI = 0.4999;
    end
    pI_ = 1 - pI;
    s_ = 1 - s; % ��������� ��������������� �����������
    G1 = zeros(1, n);
    G2 = zeros(1, n);
    
    % 2.1. ���������� ������ �������� ������� 
    ns = sum(abs(s(:, 1) - s(:, 2))); % ����� ���������� ������������� ���������
    l0_ = log(pw(2) / pw(1)); % ����� �������� �������
    L0  = log(pw(2) / pw(1)) / (2 * log(pI_) - 2 * log(pI)) + ns / 2; 
	L0r = floor(L0);
    
    % 2.2. ����� ������ ���������� ������������� ����������� �������,
    % ������� ����� ������������� ���������
    mask10 = (s(:, 1)==1)&(s(:, 2)==0);   % ����� ������������� ���������
    mask01 = (s(:, 1)==0)&(s(:, 2)==1);
    
    % 2.3. ����������� ������������ ������������� 
    if pI < 0.5 % ������ ������������ ������ 
        p12th(t) = binocdf(L0r, ns, 1 - pI); 
		p21th(t) = 1 - binocdf(L0r, ns, pI);
    else
        p12th(t) = 1 - binocdf(L0r, ns, 1 - pI);
		p21th(t) = binocdf(L0r, ns, pI);
    end
    
    % 3.������������ ��������� ������� �������������� ���������
    for kk = 1 : K % ���� �� ����� ����������
        for i = 1 : M % ���� �� �������
            % 3.1. ������������� ���������
            x = s(:, i);
            r = rand(n, 1); 
			ir = find(r < pI);
            x(ir) = 1 - x(ir); % ��������� ��������� � �������� � ��������� ������
            x_ = 1 - x;
            
            % ����� ���������� ����� � ��������������
            % 3.2. ������������� ����������� ������
            L10 = sum(x(mask10));
            P01 = sum(x_(mask01));
            u = ((L10 + P01) - L0); % ���������� �������� ����������� �������
            if pI < 0.5 
                if u>0
                    iai = 1;
                else
                    iai = 2;
                end
            else % ������� ������� �������� �������
                if u>0
                    iai = 2;
                else
                    iai = 1;
                end
            end
            
            % 3.3. �������� ���������� �������������
            Pc_(i, iai, t) = Pc_(i, iai, t) + 1; % �������� ���������� �������������
            if (kk == 1) && (t == 2) % ����������� �������� ��������� ��������
                IAx = reshape(x_, 5, 7)'; 
                figure;
                imshow(IAx);
            end
        end
    end
    Pc_(:, :, t) = Pc_(:, :, t) / K;
    p12ex(t) = Pc_(1, 2, t);
    p21ex(t) = Pc_(2, 1, t);
end

%% 4. ������������ ���������� � ���� �������� ������������ ������
figure;
grid on;
hold on;
ms = 1;
axis([min(pi), max(pi), 0, ms]); % ��������� ������ ���� ������� �� ����
p = plot(pi, p12th, '-b', pi, p21th, '-r', pi, p12ex, '--ok', pi, p21ex, '--^k');
set(p, 'LineWidth', 1.0);
title('������������� ����������� ������ � �� ������', 'FontName', 'Courier', 'FontSize', 14);
xlabel('pi', 'FontName', 'Courier', 'FontSize', 14);
ylabel('P', 'FontName', 'Courier', 'FontSize', 14);
strv1 = ' pw='; 
strv2 = num2str(pw, '% G');
text(0.1, 0.75 * ms, [strv1, strv2], 'HorizontalAlignment', 'left', 'BackgroundColor',[.8 .8 .8], 'FontSize', 12);
legend('p12th ', 'p21th', 'p12ex', 'p21ex'); 
hold off;




