from prefixBorderArray import get_prefix_border_array, get_prefix_border_array_m

def kmp(p, t):
  result = []
  bp = get_prefix_border_array(p)
  m = len(p)
  n = len(t)
  bpm = get_prefix_border_array_m(bp)
  k = 0

  for i in range(n):
    while k > 0 and p[k] != t[i]:
      k = bpm[k - 1]
    if p[k] == t[i]:
      k += 1
    if k == m:
      result.append(i - k + 1)
      k = bpm[k - 1]
  return result
