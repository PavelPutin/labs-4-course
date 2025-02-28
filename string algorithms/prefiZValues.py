def str_comp(s, i1, i2):
  n = len(s)
  eq_len = 0
  while i1 < n and i2 < n and s[i1] == s[i2]:
    i1 += 1
    i2 += 1
    eq_len += 1
  return eq_len

def str_comp_back(s, i1, i2):
  eq_len = 0
  while i1 >= 0 and i2 >= 0 and s[i1] == s[i2]:
    i1 += 1
    i2 += 1
    eq_len += 1
  return eq_len

def get_prefix_z_values(s):
  n = len(s)
  zp = [0] * n
  l = 0
  r = 0
  for i in range(1, n):
    if i >= r:
      zp[i] = str_comp(s, 0, i)
      l = i
      r = l + zp[i]
    else:
      j = i - l
      if zp[j] < r - i:
        zp[i] = zp[j]
      else:
        zp[i] = r - i + str_comp(s, r - i, r)
        l = i
        r = l + zp[i]
  return zp

def get_suffix_z_values(s):
  n = len(s)
  zs = [0] * n
  l = n - 1
  r = n - 1
  for i in range(n - 2, -1, -1):
    if i <= l:
      zs[i] = str_comp_back(s, i, n - 1)
      r = i
      l = r - zs[i]
    else:
      j = n - (r + 1 - i)
      if zs[j] < i - l:
        zs[i] = zs[j]
      else:
        zs[i] = i - l + str_comp_back(s, l, n - i + l)
        r = i
        l = r - zs[i]
  return zs

def zp_to_bpm(zp):
  n = len(zp)
  bpm = [0] * n
  for j in range(n - 1, -1, -1):
    i = j + zp[j] - 1
    bpm[i] = zp[j]
  return bpm

def zp_to_bp(zp):
  n = len(zp)
  bp = [0] * n
  for j in range(1, n):
    for i in range(j + zp[j] - 1, j - 1, -1):
      if bp[i] > 0:
        break
      bp[i] = i - j + 1
  return bp

def val_grow(n_arr, n_pos, n_val):
  n = len(n_arr)
  n_seq_len = 0
  while n_pos < n and n_arr[n_pos] == n_val:
    n_pos += 1
    n_val += 1
    n_seq_len += 1
  return n_seq_len

def bp_to_zp(bp):
  n = len(bp)
  l = 0
  r = 0
  zp = [0] * n
  for i in range(1, n):
    if i >= r:
      zp[i] = val_grow(bp, i, 1)
      l = i
      r = l + zp[i]
    else:
      j = i - l
      if zp[j] < r - i:
        zp[i] = zp[j]
      else:
        zp[i] = r - i + val_grow(bp, r, r - i + 1)
        l = i
        r = l + zp[i]
  return zp