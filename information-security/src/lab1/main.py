import sys
from encryption.bit_utils import *
from encryption.block_utils import *

K = 0x1234567890abcdef


def get_round_key(i):
    k_i = shift_right(K, i * 8)
    return k_i


def f(b, k_i):
    a = shift_left(b, 9)
    b1 = shift_right(k_i, 11)
    c = b1 & b
    c_inv = invert(c)
    return a ^ c_inv


def shipher_round(integers, i):
    k_i = get_round_key(i)
    for j in range(0, len(integers), 2):
        l = integers[j]
        r = integers[j + 1]

        lf = f(l, k_i)
        r = lf ^ r

        integers[j] = r
        integers[j + 1] = l
    return integers


def last_round_deshuffle(integers):
    for j in range(0, len(integers), 2):
        l = integers[j]
        r = integers[j + 1]
        integers[j] = r
        integers[j + 1] = l
    return integers


def encode(byte_data):
    blocks = get_blocks(byte_data)
    integers = blocks_to_integers(blocks)
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
