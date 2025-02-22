function u = systemeqv(a, b)
% распределение Коши с параметрами масштаба и формы a, b
u = a + b * tan(2 * pi * rand());