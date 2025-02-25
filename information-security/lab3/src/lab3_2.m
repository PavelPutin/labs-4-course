%% Стратегическое и тактическое планирование модельного эксперимента
clear all;

% задание количества факторов и диапазонов значений факторов
nf = 1;
M = 369;
p = 10007;
%       r_0
minf = [0];
maxf = [1];

% задание количества уровней каждого фактора
level = [10000];

% формирование полного плана эксперимента
fulplan = fullfact(level);

% формирование массив исходных данных – значений уровней факторов в диапазонах выбранных значений, которые реально будут
% использоваться в ходе эксперимента
N = prod(level) %#ok<NOPTS>
fuleks = zeros(N, nf);
for i = 1 : nf
    for j = 1 : N
        fuleks(j, i) = round(minf(i) + (fulplan(j, i) - 1) * (maxf(i) - minf(i)) / (level(i) - 1));
    end
end

% цикл по совокупности экспериментов стратегического плана

u = zeros(1, p - 1);
u_i = 1;
for i = 1 : p - 1
    [u_i, r(i)] = systemeqv(u_i, M, p);
end
test2_1(r, p - 1);

% for j = 1 : N
%     if (fuleks(j, 1) >= fuleks(j, 2))
%         continue;
%     end
% 
%     u_i(end + 1) = fuleks(j, 1);
%     p(end + 1) = fuleks(j, 2);
%     M(end + 1) = fuleks(j, 3);
%     [L_s(end + 1), l_s(end + 1)] = test1(u_i(end), M(end), p(end));
% end
