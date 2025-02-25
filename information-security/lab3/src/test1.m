function [L, l] = test1(u_i, M, p)
%TEST1 Summary of this function goes here
%   Detailed explanation goes here

% Ищем длину периода l
i_l = 15000;
i = 0;
R_i_l = 0;
u_i_l = u_i;
while true
    [u_i_l, r] = systemeqv(u_i_l, M, p);
    i = i + 1;
    if i == i_l
        R_i_l = r;
    end
    if i > i_l && r == R_i_l
        l = i - i_l;
        break;
    end
end

i = 0;
u_i_L1 = u_i;
[u_i_L1, R1] = systemeqv(u_i_L1, M, p);
u_i_L2 = u_i;
while true
    i = i + 1;
    [u_i_L1, R1] = systemeqv(u_i_L1, M, p);
    if i >= l
        [u_i_L2, R2] = systemeqv(u_i_L2, M, p);
        % [R1, R2]
        if R1 == R2
            L = i;
            break;
        end
    end
end

