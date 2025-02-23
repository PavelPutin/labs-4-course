N = 32

def shift_right(n, d):
  return ((n >> d) % (1 << N)) | (((n & (0xffff_ffff >> abs(N - d)) % (1 << N)) << abs(N - d)) % (1 << N))

def shift_left(n, d):
  return ((n << d) % (1 << N)) | (((n & (0xffff_ffff << abs(N - d)) % (1 << N)) >> abs(N - d)) % (1 << N))

def invert(i):
  return (i ^ 0xffff_ffff) % (1 << N)

def get_left_32_from_64(n):
  return (n >> N) % (1 << N)

def get_right_32_from_64(n):
  return n % (1 << N)