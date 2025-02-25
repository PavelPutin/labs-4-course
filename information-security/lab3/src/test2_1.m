function [] = test2_1(random_values, L)
N = L;
M = 2000;
alpha = 0.001;

centers = linspace(0, 1, M);
[m, centers] = hist(random_values, centers);
m = (m - (N / M)) .^ 2 ./ (N / M);
chi = sum(m);
chi_critical = chi2inv(1 - alpha, M - 2);
if chi > chi_critical
    disp("Возможно, неравномерное распределение");
end
histogram(random_values, centers);
end

