function u = systemeqv(a, b)
% распределение Коши с параметрами масштаба и формы a, b
u = tan(2 * pi * (a + b * rand()));