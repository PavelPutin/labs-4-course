import string

def position_list(s, a):
  n_a = len(a)
  m = len(s)
  pl = [[] for _ in range(n_a)]

  for k in range(m - 1, -1, -1):
    ich = s[k] - a[0]
    pl[ich].append(k)
  return pl

def bad_char_shift(pl, char_bad, pos_bad):
  if pos_bad < 0:
    return 1
  n_pos = -1
  l = pl[char_bad]
  if len(l) != 0:
    n_len = len(l)
    for k in range(n_len):
      if l[k] < pos_bad:
        n_pos = l[k]
        break
  return pos_bad - n_pos

def bm(p, t):
  result = []
  a = string.ascii_letters
  pl = position_list(p, a)
  m = len(p)
  n = len(t)
  n_text_r = m

  while n_text_r <= n:
    k = m - 1
    i = n_text_r - 1
    while k >= 0:
      if p[k] != t[i]:
        break
      k -= 1
      i -= 1
    if k < 0:
      result.append(k + 1)
    n_text_r += bad_char_shift(pl, t[i], k)
  return result