def gorner_mod_2(s, m, q):
  res = 0
  for i in range(m):
    res = (res * 2 + int(s[i])) % q
  return res


def kr(p, t, q):
  result = []
  p = [int(i) for i in p]
  t = [int(i) for i in t]
  n = len(t)
  m = len(p)
  p2m = 1
  for _ in range(m - 1):
    p2m = (p2m * 2) % q
  hp = gorner_mod_2(p, m, q)
  ht = gorner_mod_2(t, m, q)

  for j in range(n - m + 1):
    if ht == hp:
      k = 0
      while k < m and p[k] == t[j + k]:
        k += 1
      if k == m:
        result.append(j)
    if j + m >= n:
      break
    ht = ((ht - p2m * t[j]) * 2 + t[j + m]) % q
    if ht < 0:
      ht += q
  return result


print(kr("0110", "1100110010011001", 1000))