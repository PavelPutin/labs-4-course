clear all;

Ts = 60 * 24 * 7; % моделирование 7 дней работы
s = sim('my_queueing', Ts); % моделирование
disp('Среднее число заявок в системе:');
disp(mean(s.inSystem.Data));

p = s.success.Data(end) ./ (s.success.Data(end) + s.failure.Data(end));
disp('Вероятность обслуживания клиента:');
disp(p);

disp('Пропускная способность системы за неделю:');
disp(s.totalUsers.Data(end));