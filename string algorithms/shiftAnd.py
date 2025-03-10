import string

def shift_and(p, t):
  result = []
  m = len(p)
  n = len(t)
  a = string.ascii_letters + string.digits

  b = {ch: 0 for ch in a}
  for i in range(m):
    b[p[i]] |= 1 << (m - 1 - i)
  u_high = 1 << (m - 1)
  M = 0
  for i in range(n):
    M = (M >> 1 | u_high) & b[t[i]]
    if M & 1:
      result.append(i - m + 1)
  return result


def shift_and_fz(p, t, k):
  """
  Поиск всех вхождений образца P длины m в текст T длины n
  При этом допускается несовпадение до k < m символов
  """
  result = []
  m = len(p)
  n = len(t)
  a = string.ascii_letters + string.digits

  b = {ch: 0 for ch in a}
  for i in range(m):
    b[p[i]] |= 1 << (m - 1 - i)
  u_high = 1 << (m - 1)

  M = [0 for _ in range(k + 1)]
  M1 = [0 for _ in range(k + 1)]

  for i in range(n):
    for l in range(k + 1):
      M1[l] = M[l]
      M[l] = (M[l] >> 1 | u_high) & b[t[i]]
      if l != 0:
        M[l] |= (M1[l - 1] >> 1 | u_high)
      if l == k and M[l] & 1:
        result.append(i - m + 1)
  return result


print(shift_and_fz('aaabaa', 'aaacaabaaabaaabaaa', 1))