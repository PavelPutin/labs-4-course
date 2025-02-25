function [u, r] = systemeqv(u_i, M, p)
%SYSTEMEQV Генератор случайных чисел Лемера
% $U_{i+1} = (U_i * M) mod p$
% $U_i, M, p$ - целые числа
% u - следующее значение u, $1 <= u_i <= p - 1$
% r - случайное число, $0 <= r <= r$
u = mod((u_i * M), p);
r = u_i / p;
end

