% ���� pr54_rec_bin. ������ � ������ ���������� ������������� ������� �� ���������� 
% ��������� (�� ������� ������������� �������� �����������)

% ������ ��� �Ш� ������� � ������� ������ ������

clear all;
close all;
%% 1. ������� �������� ������
n = 35; % ���������� ��������� (������ �� ������� �����������)
M = 3; % ����� � �������� �� 3
s = zeros(n, M); % ���������� ������� � ��������� ��������
% 1.1. ������� �������� ������� (����� ����������� �������� ������ �����)
letterP = [1 1 1 1 1 ...   
           1 0 0 0 1 ...
           1 0 0 0 1 ...
           1 0 0 0 1 ...
           1 0 0 0 1 ...
           1 0 0 0 1 ...
           1 0 0 0 1 ]';
letterY = [1 0 0 0 1 ...   
           1 0 0 0 1 ...
           1 0 0 0 1 ...
           1 1 1 1 1 ...
           0 0 0 0 1 ...
           0 0 0 0 1 ...
           1 1 1 1 1 ]';
letterA = [1 1 1 1 1 ...   % ����������� ������ �����
           1 0 0 0 1 ...
           1 0 0 0 1 ...
           1 0 0 0 1 ...
           1 1 1 1 1 ...
           1 0 0 0 1 ...
           1 0 0 0 1 ]';        
s(:, 1) = letterP; 
s(:, 2) = letterY;
s(:, 3) = letterA; % ����������� ������ �����

% pw1 < pw2 < pw3
% pw1 < pw2 = pw3
% pw1 < pw2 > pw3
% pw1 = pw2 < pw3
% pw1 = pw2 = pw3
% pw1 = pw2 > pw3
% pw1 > pw2 < pw3
% pw1 > pw2 = pw3
% pw1 > pw2 > pw3
PW = [0.2 0.3 0.5; % pw1 < pw2 < pw3
      0.2 0.4 0.4; % pw1 < pw2 = pw3
      0.2 0.5 0.3; % pw1 < pw2 > pw3
      0.3 0.3 0.4; % pw1 = pw2 < pw3
      0.333333 0.333333 0.333334; % pw1 = pw2 = pw3
      0.4 0.4 0.2; % pw1 = pw2 > pw3
      0.4 0.2 0.3; % pw1 > pw2 < pw3
      0.4 0.3 0.3; % pw1 > pw2 = pw3
      0.5 0.3 0.2];% pw1 > pw2 > pw3
% 1.2. ������� ���������� ������������ (����� ����������� ������ �������� � pw)
for pwi = 1:9
    pw = PW(pwi,:);
    disp('��������� ����������� �������');
    disp(pw);
    % pw = [0.5 0.5 0.5]; %��������� ����������� �������
    np = sum(pw);
    pw = pw / np; % ���������� ������������� ������� ��������� ������������
    N = 20;  % ���������� ����� ��������� ������������ ��������� - pi
    K = 1000; % ���������� ����������
    pi = zeros(1, N); % ������ ������������ ��������� �������
    
    % ����� ��������� ����� 1.3.
    
    % 1.4. ������� ������ (����� ����������� ������������� ������� ������)
    Pc_ = zeros(M); % ����������������� ������� ������������ ������
    Pt = zeros(M); % ������� ������ ���������
    
    % ����� ��������� ���������� ����� (���� ����� �������)
    % ���� �� ��������� ����������� ��������� ��������� ��������
    % for t = 1 : N    
    %     pi(t) = (1 / N) * (t - 1);
    %     pI = pi(t); % ����������� ��������� �������� (�������)
    pI = 0.6; % ����� �������� ������������� �������� ����������� ��������� (pI)
    
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
    
    % ����������� ������� ���� � ������ 1.1. - 1.3. ������ ������ ����
    % � �.1.1-1.2 ������� ��� s() � ��� pw() ������ 1 �������� �� ii,
    % ������ 2 �������� �� jj
    % �. 1.1. � 1.2. ���� ����������� � �.3.2.
    % �������� ��������� �������
    for ii = 1 : M - 1
        for jj = ii + 1 : M
            % 2.1. ���������� ������ �������� ������� 
            ns = sum(abs(s(:, ii) - s(:, jj))); % ����� ���������� ������������� ���������
            l0_ = log(pw(jj) / pw(ii)); % ����� �������� �������
            L0  = log(pw(jj) / pw(ii)) / (2 * log(pI_) - 2 * log(pI)) + ns / 2; 
            L0r = floor(L0);
            
            % 2.2.���������� ������������� ����������� ������� (���� ����� ������ ����� ������)
            for k = 1 : n
               G1(1, k) = log((s(k, ii) * pI_ + s_(k, ii) * pI) / (s(k, jj) * pI_ + s_(k, jj) * pI));
               G2(1, k) = log((s(k, ii) * pI + s_(k, ii) * pI_) / (s(k, jj) * pI + s_(k, jj) * pI_));
            end
            
            % 2.3. ����������� ������������ �������������
            % ����� ������ p12th(t) ����� Pt(ii,jj), ������ p21th(t) - Pt(jj,ii)
            if pI < 0.5 % ������ ������������ ������
                Pt(ii,jj) = binocdf(L0r, ns, 1-pI);
                Pt(jj,ii) = 1 - binocdf(L0r, ns, pI);
            else
                Pt(ii,jj) = 1 - binocdf(L0r, ns, 1 - pI);
                Pt(jj,ii) = binocdf(L0r, ns, pI);
            end
        end
    end
    % ����� ����������� ������� � ����������� ������������ ��������� ����. ������� ������
    % ���������� ������������ ����������� ������������� (������������ ��������)
    Pt = Pt + diag(ones(3, 1) - sum(Pt, 2));
    
    %% 3. ������������ ��������� ������� �������������� ���������
    for kk = 1 : K % ���� �� ����� ����������
        for i = 1 : M % ���� �� �������
            % 3.1. ������������� ���������
            x = s(:, i);
            r = rand(n, 1); 
            ir = find(r < pI);
            x(ir) = 1 - x(ir); % ��������� ��������� � �������� � ��������� ������
            x_ = 1 - x;
            
            % 3.2. ������������� ����������� ������ (�������� ��������� �������)
            % ����� ����������� ������������� ������� ��� ��������
            % ����������� �������� ��������� �������; �.3.2 ����������
            % ������ �������� ����� ������ � �.2.1 � 2.2.
            iais = [];  % ���������� �������� ��������� (������� �������)
            for ii = 1 : M - 1
                for jj = ii + 1 : M
                    % ����� 2.1. ���������� ������ �������� ������� 
                    ns = sum(abs(s(:, ii) - s(:, jj))); % ����� ���������� ������������� ���������
                    l0_ = log(pw(jj) / pw(ii)); % ����� �������� �������
                    L0  = log(pw(jj) / pw(ii)) / (2 * log(pI_) - 2 * log(pI)) + ns / 2;
                    L0r = floor(L0);
    
                    % ����� 2.2.���������� ������������� ����������� �������
                    for k = 1 : n % ���������� ������������� ����������� �������
                       G1(1, k) = log((s(k, ii) * pI_ + s_(k, ii) * pI) / (s(k, jj) * pI_ + s_(k, jj) * pI));
                       G2(1, k) = log((s(k, ii) * pI + s_(k, ii) * pI_) / (s(k, jj) * pI + s_(k, jj) * pI_));
                    end
                    
                    % ����� ������ �������� ������� ����� ii � jj
                    % 3.2. ������������� ����������� ������
                    u = G1 * x + G2 * x_ - l0_; % ���������� �������� ����������� �������
                    if u > 0
                        iai = ii;
                    else
                        iai = jj;
                    end
                    
                    % ����� ���������� ��������� ��������� ���������
                    iais = [iais, iai];
                end
            end
            % ����� �������� ������ ������, �� ������� �������������
            % ����������� ������ ���������������
            id = mode(iais);   % ����� ����� ������������� ������
            
            % 3.3. �������� ���������� �������������
            Pc_(i, id) = Pc_(i, id) + 1; % �������� ���������� �������������            
        end
    end
    Pc_ = Pc_ / K;  % ����� ����� ������ ������� ��� Pc_
    % end;  % ��������� end ����������� � ����� �� t
    % ����������� ����� ������
    disp('������������� ������� ������')
    disp(Pt)
    disp('����������������� ������� ������')
    disp(Pc_)
end