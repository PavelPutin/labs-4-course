import sys

N = 32
BLOCK_SIZE = 8
K = 0xa234567890abcdef
verbose = False

def shift_right(n, d):
  return ((n >> d) % (1 << N)) |( ((n & (0xffff_ffff >> abs(N - d)) % (1 << N)) << abs(N - d)) % (1 << N))

def shift_left(n, d):
  return ((n << d) % (1 << N)) | (((n & (0xffff_ffff << abs(N - d)) % (1 << N)) >> abs(N - d)) % (1 << N))

def get_blocks(byte_data):
  blocks = [byte_data[i:i + BLOCK_SIZE] for i in range(0, len(byte_data), BLOCK_SIZE)]
  if len(blocks[-1]) != BLOCK_SIZE:
    blocks[-1] += b'0' * (BLOCK_SIZE - len(blocks[-1]))
  return blocks

def blocks_to_integers(blocks):
  integers = []
  for block in blocks:
    integers.append(int.from_bytes(block[:BLOCK_SIZE // 2]))
    integers.append(int.from_bytes(block[BLOCK_SIZE // 2:]))
  return integers

def integers_to_blocks(integers):
  blocks = []
  for i in range(0, len(integers), 2):
    integer1 = integers[i]
    integer2 = integers[i + 1]
    l = int.to_bytes(integer1, length=BLOCK_SIZE // 2)
    r = int.to_bytes(integer2, length=BLOCK_SIZE // 2)
    blocks.append(l + r)
  return blocks

def join_blocks(blocks):
  return b''.join(blocks).rstrip(b'0')

def get_round_key(i):
  k_i = shift_right(K, i * 8)
  return k_i % (1 << N)


def invert(i):
  return i ^ 0xffff_ffff


def f(b, k_i):
  a = shift_left(b, 9)
  b1 = shift_right(k_i, 11)
  c = b1 & b
  c_inv = invert(c)
  return (a ^ c_inv) % (1 << N)

def shipher_round(integers, i):
  k_i = get_round_key(i)
  if (verbose):
    print(k_i, end=" ")
  for j in range(0, len(integers), 2):
    l = integers[j]
    r = integers[j + 1]
    
    lf = f(l, k_i)
    r = lf ^ r

    integers[j] = r
    integers[j + 1] = l
  if (verbose):
    print(integers)
  return integers

def last_round_deshuffle(integers):
  for j in range(0, len(integers), 2):
    l = integers[j]
    r = integers[j + 1]
    integers[j] = r
    integers[j + 1] = l
  if (verbose):
    print("last round:", integers)
  return integers

def encode(byte_data):
  blocks = get_blocks(byte_data)
  integers = blocks_to_integers(blocks)
  if (verbose):
    print("start encode", integers)
  for i in range(8):
    integers = shipher_round(integers, i)
  else:
    integers = last_round_deshuffle(integers)
  blocks = integers_to_blocks(integers)
  result = join_blocks(blocks)
  return result


def decode(byte_data):
  blocks = get_blocks(byte_data)
  integers = blocks_to_integers(blocks)
  if (verbose):
    print("start decode", integers)
  for i in range(7, -1, -1):
    integers = shipher_round(integers, i)
  else:
    integers = last_round_deshuffle(integers)
  blocks = integers_to_blocks(integers)
  result = join_blocks(blocks)
  return result

if __name__ == "__main__":
  key_path = sys.argv[1]
  with open(key_path, "rb") as key_file:
    K = key_file.read()
    K = int.from_bytes(K)
  encoding = sys.argv[2] == "encode"
  target_path = sys.argv[3]
  with open(target_path, "rb") as target_file:
    byte_data = target_file.read()

  result = encode(byte_data) if encoding else decode(byte_data)

  output_file_name = target_path + ".encoded" if (encoding) else target_path + ".decoded"
  with open(output_file_name, "wb") as output_file:
    output_file.write(result)