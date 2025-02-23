import sys
from encryption.bit_utils import *
from encryption.block_utils import *


def get_round_key(K, i):
    return shift_right(K, i * 8)


def f_custom(b, k_i):
    a = shift_left(b, 9)
    b1 = shift_right(k_i, 11)
    c = b1 & b
    c_inv = invert(c)
    return a ^ c_inv


def apply_rounds(block, keys, f):
    print("keys:", list(keys))
    print("initial:", block)
    for key in keys:
        block = shipher_round(block, key, f)
        block.reverse()
        print(block)
    block.reverse()
    print("result:", block)
    print()
    return block


def shipher_round(block, key, f):
    l, r = block
    lf = f(l, key)
    print("lf:", lf, end=" ")
    r = lf ^ r
    return [l, r]


class ECB:
    def __init__(self, key, rounds, f):
        self.key = key
        self.rounds = rounds
        self.f = f
        self.keys = [get_round_key(self.key, i) for i in range(rounds)]

    def encode(self, byte_data):
        print("encode")
        blocks = get_blocks(byte_data)
        print("blocks: ", blocks)
        integers = blocks_to_integers(blocks)
        print("integers: ", integers)
        for i in range(0, len(integers), 2):
            block = [integers[i], integers[i + 1]]
            block = apply_rounds(block, self.keys, self.f)
            integers[i], integers[i + 1] = block
        blocks = integers_to_blocks(integers)
        return join_blocks(blocks)

    def decode(self, byte_data):
        print("decode")
        blocks = get_blocks(byte_data)
        print("blocks: ", blocks)
        integers = blocks_to_integers(blocks)
        print("integers: ", integers)
        for i in range(0, len(integers), 2):
            block = [integers[i], integers[i + 1]]
            block = apply_rounds(block, list(reversed(self.keys)), self.f)
            integers[i], integers[i + 1] = block
        blocks = integers_to_blocks(integers)
        return join_blocks(blocks)


if __name__ == "__main__":
    key_path = sys.argv[1]
    with open(key_path, "rb") as key_file:
        K = key_file.read()
        K = int.from_bytes(K)
    encoding = sys.argv[2] == "encode"
    target_path = sys.argv[3]
    with open(target_path, "rb") as target_file:
        byte_data = target_file.read()

    ecb = ECB(K, 8, f_custom)
    result = ecb.encode(byte_data) if encoding else ecb.decode(byte_data)

    output_file_name = target_path + ".encoded" if encoding else target_path + ".decoded"
    with open(output_file_name, "wb") as output_file:
        output_file.write(result)
