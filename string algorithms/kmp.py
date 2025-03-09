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

def online_kmp(p, t):
  result = []
  bp = get_prefix_border_array(p)
  m = len(p)
  n = len(t)
  bpm = get_prefix_border_array_m(bp)
  a = set(list(t))
  bpm2 = {letter: [-1] * m for letter in a}
  for letter in a:
    for i in range(m):
      bpm2[letter][i] = bpm[i] + 1 if bpm[i] > 0 and letter == p[bpm[i]] else -1
  k = 0
  for i in range(n):
    letter = t[i]
    
    if letter != p[k]:
      k = bpm2[letter][k - 1] + 1
    else:
      k += 1
    if k == m:
      result.append(i - k + 1)
      k = bpm[k - 1]
  return result


print(online_kmp('aaabaa', 'aaacaabaaabaaabaaa'))