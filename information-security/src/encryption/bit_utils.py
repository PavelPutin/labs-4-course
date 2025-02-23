N = 32

def shift_right(n, d):
  return ((n >> d) % (1 << N)) | (((n & (0xffff_ffff >> abs(N - d)) % (1 << N)) << abs(N - d)) % (1 << N))

def shift_left(n, d):
  return ((n << d) % (1 << N)) | (((n & (0xffff_ffff << abs(N - d)) % (1 << N)) >> abs(N - d)) % (1 << N))

def invert(i):
  return (i ^ 0xffff_ffff) % (1 << N)