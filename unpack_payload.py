import numpy as np

all_bins = [
    1792, 184, 528, 1444, 1288, 60, 1980, 1684, # header
    1536, 872, 1600, 1472, 456, 368, 1736, 688, # block 0
    1772, 860, 1660, 1616, 428, 888, 728, 1512, # block 1
    1724, 1396, 1712, 1216, 672, 700, 104, 1756, # block 2
    1012, 484, 124, 1628, 300, 1932, 388, 268,  # block 3
    628, 956, 436, 508, 1236, 1156, 548, 516,   # block 4
    620, 180, 1356, 1552, 908, 1504, 1984, 1344, # block 5
    1252, 1584, 248, 1960, 1628, 936, 636, 356,  # block 6
    948, 432, 784, 1060, 988, 396, 592, 976     # block 7
]

def rotl(val, shift, bits):
    return ((val << shift) | (val >> (bits - shift))) & ((1 << bits) - 1)

def deinterleave(words, ppm=9):
    bits_per_word = len(words)
    words_deint = [0] * ppm
    for i in range(bits_per_word):
        word = rotl(words[i], i, ppm)
        for x in range(ppm):
            bit = (word >> x) & 1
            words_deint[x] |= (bit << i)
    return words_deint

shuffle_pattern = [5, 0, 1, 2, 4, 3, 6, 7]

def deshuffle(words):
    res = []
    for w in words:
        r = 0
        for j, s in enumerate(shuffle_pattern):
            r |= ((w >> s) & 1) << j
        res.append(r)
    return res

b0 = 16
ppm = 9

all_codewords = []
for blk in range(8):
    blk_bins = all_bins[8 + blk*8 : 8 + (blk+1)*8]
    bin_idxs = [((b - b0) % 2048) // 4 for b in blk_bins]
    words_gr = [x ^ (x >> 1) for x in bin_idxs]
    deint_gr = deinterleave(words_gr, ppm)
    deshuf_gr = deshuffle(deint_gr)
    all_codewords.extend(deshuf_gr)

print(f"Total codewords: {len(all_codewords)}")
# Codewords are 8 bits.
# High nibble is data nibble:
nibbles_high = [cw >> 4 for cw in all_codewords]
# What if low nibble is data nibble?
nibbles_low = [cw & 0xF for cw in all_codewords]

print("Nibbles high (72):", nibbles_high)
print("Nibbles low  (72):", nibbles_low)

# Let's pack nibbles into 36 bytes:
# Option 1: nibbles_high, (n0 << 4) | n1
bytes_h1 = bytes([(nibbles_high[i*2] << 4) | nibbles_high[i*2+1] for i in range(36)])
# Option 2: nibbles_high, (n1 << 4) | n0
bytes_h2 = bytes([(nibbles_high[i*2+1] << 4) | nibbles_high[i*2] for i in range(36)])
# Option 3: nibbles_low, (n0 << 4) | n1
bytes_l1 = bytes([(nibbles_low[i*2] << 4) | nibbles_low[i*2+1] for i in range(36)])
# Option 4: nibbles_low, (n1 << 4) | n0
bytes_l2 = bytes([(nibbles_low[i*2+1] << 4) | nibbles_low[i*2] for i in range(36)])

print("Raw bytes H1:", bytes_h1)
print("Raw bytes H2:", bytes_h2)
print("Raw bytes L1:", bytes_l1)
print("Raw bytes L2:", bytes_l2)
