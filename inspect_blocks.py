import numpy as np

# Let's inspect the raw payload symbols
# We had:
# 8 header symbols (Sym 0 to 7)
# 64 payload symbols (Sym 8 to 71)

# All 72 extracted bins from extract_decimated.py:
all_bins = [
    1792, 184, 528, 1444, 1288, 60, 1980, 1684, # header (0..7)
    1536, 872, 1600, 1472, 456, 368, 1736, 688, # block 0 (8..15)
    1772, 860, 1660, 1616, 428, 888, 728, 1512, # block 1 (16..23)
    1724, 1396, 1712, 1216, 672, 700, 104, 1756, # block 2 (24..31)
    1012, 484, 124, 1628, 300, 1932, 388, 268,  # block 3 (32..39)
    628, 956, 436, 508, 1236, 1156, 548, 516,   # block 4 (40..47)
    620, 180, 1356, 1552, 908, 1504, 1984, 1344, # block 5 (48..55)
    1252, 1584, 248, 1960, 1628, 936, 636, 356,  # block 6 (56..63)
    948, 432, 784, 1060, 988, 396, 592, 976     # block 7 (64..71)
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

def gray_dec(g):
    b = g
    while g > 0:
        g >>= 1
        b ^= g
    return b

# Check for b0 = 16 (or around 16)
b0 = 16
ppm = 9

# Let's inspect block by block
for blk in range(8):
    blk_bins = all_bins[8 + blk*8 : 8 + (blk+1)*8]
    bin_idxs = [((b - b0) % 2048) // 4 for b in blk_bins]
    
    # Try different mappings:
    # 1. Gray decode:
    words_gray = [gray_dec(x) for x in bin_idxs]
    # 2. gr-lora gray:
    words_gr = [x ^ (x >> 1) for x in bin_idxs]
    # 3. raw:
    words_raw = list(bin_idxs)
    
    deint_gray = deinterleave(words_gray, ppm)
    deshuf_gray = deshuffle(deint_gray)
    
    deint_gr = deinterleave(words_gr, ppm)
    deshuf_gr = deshuffle(deint_gr)
    
    print(f"Block {blk}:")
    print(f"  Gray deshuf: {[hex(x) for x in deshuf_gray]}")
    print(f"  GR   deshuf: {[hex(x) for x in deshuf_gr]}")
