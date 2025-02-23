import logging
from abc import ABC, abstractmethod
from encryption.bit_utils import *
from encryption.block_utils import *


def f_custom(b, k_i):
    a = shift_left(b, 9)
    b1 = shift_right(k_i, 11)
    c = b1 & b
    c_inv = invert(c)
    return a ^ c_inv


class BaseCypherAlgo(ABC):
    def __init__(self, key, rounds, f=f_custom):
        self.logger = logging.getLogger(str(self))
        self.key = key
        self.rounds = rounds
        self.f = f
        self.keys = [self._get_round_key(i) for i in range(rounds)]

    def encode(self, byte_data):
        self.logger.debug("encode")
        blocks = get_blocks(byte_data)
        self.logger.debug(f"blocks: {blocks}")
        integers = blocks_to_integers(blocks)
        self.logger.debug(f"integers: {integers}")
        for i in range(0, len(integers), 2):
            block = [integers[i], integers[i + 1]]
            block = self._encode_block(block)
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
            block = self._decode_block(block)
            integers[i], integers[i + 1] = block
        blocks = integers_to_blocks(integers)
        return join_blocks(blocks)

    @abstractmethod
    def _encode_block(self, block):
        pass

    @abstractmethod
    def _decode_block(self, block):
        pass

    def _apply_rounds(self, block, keys):
        self.logger.debug(f"keys: {list(keys)}")
        for key in keys:
            block = self._cypher_round(block, key)
            block.reverse()
            self.logger.debug(f"{block}")
        block.reverse()
        self.logger.debug(f"result: {block}")
        return block

    def _cypher_round(self, block, key):
        l, r = block
        lf = self.f(l, key)
        self.logger.debug(f"lf: {lf}")
        r = lf ^ r
        return [l, r]

    def _get_round_key(self, i):
        return shift_right(self.key, i * 8)


class ECB(BaseCypherAlgo):
    def __init__(self, key, rounds, f=f_custom):
        super().__init__(key, rounds, f)

    def _encode_block(self, block):
        return self._apply_rounds(block, self.keys)

    def _decode_block(self, block):
        return self._apply_rounds(block, list(reversed(self.keys)))


def _re_init(f):
    def wrapper(self, *args, **kwargs):
        self.prev_block = [self.init_vector[0], self.init_vector[1]]
        return f(self, *args, **kwargs)

    return wrapper


class CBC(BaseCypherAlgo):
    def __init__(self, key, init_vector, rounds, f=f_custom):
        self.init_vector = [get_left_32_from_64(init_vector), get_right_32_from_64(init_vector)]
        self.prev_block = [get_left_32_from_64(init_vector), get_right_32_from_64(init_vector)]
        super().__init__(key, rounds, f)
        self.logger.debug(f'init prev block: {self.prev_block}')

    @_re_init
    def encode(self, byte_data):
        return super().encode(byte_data)

    @_re_init
    def decode(self, byte_data):
        return super().decode(byte_data)

    def _encode_block(self, block):
        self.logger.debug(f"initial encoding: {block}")
        block = [block[i] ^ self.prev_block[i] for i in range(len(block))]
        block = self._apply_rounds(block, self.keys)
        self.prev_block = [block[i] for i in range(len(block))]
        self.logger.debug(f'prev block: {self.prev_block}')
        return block

    def _decode_block(self, block):
        temp_block = [block[0], block[1]]
        self.logger.debug(f"initial decoding: {block}")
        block = self._apply_rounds(block, list(reversed(self.keys)))
        block = [block[i] ^ self.prev_block[i] for i in range(len(block))]
        self.prev_block = [temp_block[i] for i in range(len(temp_block))]
        self.logger.debug(f'prev block: {self.prev_block}')
        return block
