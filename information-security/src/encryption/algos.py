import logging
from encryption.bit_utils import *
from encryption.block_utils import *


def f_custom(b, k_i):
    a = shift_left(b, 9)
    b1 = shift_right(k_i, 11)
    c = b1 & b
    c_inv = invert(c)
    return a ^ c_inv


class ECB:
    def __init__(self, key, rounds, f=f_custom):
        self.logger = logging.getLogger(str(self))
        self.key = key
        self.rounds = rounds
        self.f = f
        self.keys = [self.__get_round_key(i) for i in range(rounds)]

    def encode(self, byte_data):
        self.logger.debug("encode")
        blocks = get_blocks(byte_data)
        self.logger.debug(f"blocks: {blocks}")
        integers = blocks_to_integers(blocks)
        self.logger.debug(f"integers: {integers}")
        for i in range(0, len(integers), 2):
            block = [integers[i], integers[i + 1]]
            block = self.__apply_rounds(block, self.keys)
            integers[i], integers[i + 1] = block
        blocks = integers_to_blocks(integers)
        return join_blocks(blocks)

    def decode(self, byte_data):
        self.logger.debug("decode")
        blocks = get_blocks(byte_data)
        self.logger.debug(f"blocks: {blocks}")
        integers = blocks_to_integers(blocks)
        self.logger.debug(f"integers: {integers}")
        for i in range(0, len(integers), 2):
            block = [integers[i], integers[i + 1]]
            block = self.__apply_rounds(block, list(reversed(self.keys)))
            integers[i], integers[i + 1] = block
        blocks = integers_to_blocks(integers)
        return join_blocks(blocks)

    def __apply_rounds(self, block, keys):
        self.logger.debug(f"keys: {list(keys)}")
        self.logger.debug(f"initial: {block}")
        for key in keys:
            block = self.__cypher_round(block, key)
            block.reverse()
            self.logger.debug(f"{block}")
        block.reverse()
        self.logger.debug(f"result: {block}")
        return block

    def __cypher_round(self, block, key):
        l, r = block
        lf = self.f(l, key)
        self.logger.debug(f"lf: {lf}")
        r = lf ^ r
        return [l, r]

    def __get_round_key(self, i):
        return shift_right(self.key, i * 8)
