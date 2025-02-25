%% Стратегическое и тактическое планирование модельного эксперимента
clear all;

% задание количества факторов и диапазонов значений факторов
nf = 3;
%       U_i       p       M
minf = [1       10001   1];
maxf = [10007   10007   1000];

% задание количества уровней каждого фактора
level = [200 7 20];

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
L_s = zeros(0);
l_s = zeros(0);
u_i = zeros(0);
p = zeros(0);
M = zeros(0);
for j = 1 : N
    if (fuleks(j, 1) >= fuleks(j, 2))
        continue;
    end

    u_i(end + 1) = fuleks(j, 1);
    p(end + 1) = fuleks(j, 2);
    M(end + 1) = fuleks(j, 3);
    [L_s(end + 1), l_s(end + 1)] = test1(u_i(end), M(end), p(end));
end

modelling = table();
modelling.u_i = u_i';
modelling.p = p';
modelling.M = M';
modelling.L = L_s';
modelling.l = l_s';
modelling;
writetable(modelling, "output.csv")
