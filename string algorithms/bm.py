import string
from prefixBorderArray import get_suffix_border_array, get_suffix_border_array_m

def position_list(s, a):
  n_a = len(a)
  m = len(s)
  pl = {ch: [] for ch in a}

  for k in range(m - 1, -1, -1):
    pl[a[k]].append(k)
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


def bs_to_ns(bs):
  m = len(bs)
  ns = [-1] * m
  for i in range(m - 1):
    if bs[i] != 0:
      k = m - bs[i] - 1
      ns[k] = i
  return ns


def bs_to_br(bs):
  m = len(bs)
  currBorder = bs[0]
  br = [0] * m
  k = 0
  while currBorder != 0:
    while k < m - currBorder:
      br[k] = currBorder
      k += 1
    currBorder = bs[k]
  while k < m:
    br[k] = 0
    k += 1
  return br


def good_suffix_shift(nsx, br, pos_bad):
  m = len(br)
  if pos_bad == m - 1:
    return 1
  if pos_bad < 0:
    return m - br[0]
  
  copy_pos = nsx[pos_bad]
  if copy_pos >= 0:
    shift = pos_bad - copy_pos + 1
  else:
    shift = m - br[pos_bad]
  return shift


def bm_suffix(p, t, h=True):
  result = []
  a = string.ascii_letters
  pl = position_list(p, a)
  m = len(p)
  n = len(t)

  bs = get_suffix_border_array(p)
  br = bs_to_br(bs)
  if h:
    bs = get_suffix_border_array_m(bs)
  ns = bs_to_ns(bs)
  nTextR = m

  while nTextR <= n:
    k = m - 1
    i = nTextR - 1
    while k >= 0:
      if p[k] != t[i]:
        break
      k -= 1
      i -= 1
    if k < 0:
      result.append(i + 1)
    nShift = max(bad_char_shift(pl, t[i], k), good_suffix_shift(ns, br, k))
    nTextR += nShift
  return result

print(bm_suffix('aaabaa', 'aaacaabaaabaaabaaa'))