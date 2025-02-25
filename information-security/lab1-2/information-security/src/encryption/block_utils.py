BLOCK_SIZE = 8


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
