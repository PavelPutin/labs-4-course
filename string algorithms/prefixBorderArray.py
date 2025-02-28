def get_prefix_border_array(s: str) -> list[int]:
  n = len(s)
  bp = [0] * n
  for i in range(1, n):
    bp_right = bp[i - 1]
    while bp_right > 0 and s[i] != s[bp_right]:
      bp_right = bp[bp_right - 1]
    if s[i] == s[bp_right]:
      bp[i] = bp_right + 1
    else:
      bp[i] = 0
  return bp


def get_suffix_border_array(s: str) -> list[int]:
  n = len(s)
  bs = [0] * n
  for i in range(n - 2, -1, -1):
    bs_left = bs[i + 1]
    while bs_left > 0 and s[i] != s[n - bs_left - 1]:
      bs_left = bs[n - bs_left]
    if s[i] == s[n - bs_left - 1]:
      bs[i] = bs_left + 1
    else:
      bs[i] = 0
  return bs

def get_prefix_border_array_m(s, bp):
  n = len(s)
  bpm = [0] * n
  bpm[-1] = bp[-1]
  for i in range(1, n):
    if bp[i] > 0 and s[bp[i]] == s[i + 1]:
      bpm[i] = bpm[bp[i] - 1]
    else:
      bpm[i] = bp[i]
  return bpm

def get_prefix_border_array_m(bp):
  n = len(bp)
  bpm = [0] * n
  bpm[-1] = bp[-1]
  for i in range(1, n):
    if bp[i] > 0 and bp[i] + 1 == bp[i + 1]:
      bpm[i] = bpm[bp[i] - 1]
    else:
      bpm[i] = bp[i]
  return bpm

def bpm_to_bp(bpm):
  n = len(bpm)
  bp = [0] * n
  bp[-1] = bpm[-1]
  for i in range(n - 2, -1, -1):
    bp[i] = max(bp[i + 1] - 1, bpm[i])
  return bp

def get_suffix_border_array_m(bs):
  n = len(bs)
  bsm = [0] * n
  bsm[0] = bs[0]
  for i in range(n - 2, -1, -1):
    if bs[i] > 0 and bs[i] + 1 == bs[i - 1]:
      bsm[i] = bsm[n - bs[i]]
    else:
      bsm[i] = bs[i]
  return bsm

def bsm_to_bs(bsm):
  n = len(bsm)
  bs = [0] * n
  bs[0] = bsm[0]
  for i in range(1, n - 1):
    bs[i] = max(bs[i - 1] - 1, bsm[i])
  return bs
