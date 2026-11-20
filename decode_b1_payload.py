import numpy as np

# Decode payload of Burst 1
# From decode_b1.py, let's grab the bins for Burst 1
# Symbols 8 to 47 (40 symbols = 5 blocks of 8 symbols)
b1_payload_bins = [
    70, 45, 66, 93, 31, 107, 66, 123,  # block 0
    86, 121, 16, 20, 24, 54, 49, 117,  # block 1
    23, 123, 62, 44, 90, 63, 65, 61,   # block 2
    50, 67, 96, 58, 35, 53, 98, 36,    # block 3
    54, 68, 53, 28, 89, 97, 42, 99     # block 4
]

b0 = 2
ppm = 7 # SF = 7

def rotl(val, shift, bits):
    shift = shift % bits
    if shift == 0:
        return val & ((1 << bits) - 1)
    return ((val << shift) | (val >> (bits - shift))) & ((1 << bits) - 1)

def deinterleave(words, ppm):
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

b1_codewords = []
for blk in range(5):
    blk_bins = b1_payload_bins[blk*8 : (blk+1)*8]
    bin_idxs = [((b - b0) % 128) for b in blk_bins]
    words_gr = [x ^ (x >> 1) for x in bin_idxs]
    deint = deinterleave(words_gr, ppm)
    deshuf = deshuffle(deint)
    print(f"B1 Block {blk}: {[hex(x) for x in deshuf]}")
    b1_codewords.extend(deshuf)

# Extract nibbles:
nibbles = [cw >> 4 for cw in b1_codewords]
print("B1 nibbles:", nibbles)

# Pack nibbles into bytes
b1_bytes1 = bytes([(nibbles[i*2] << 4) | nibbles[i*2+1] for i in range(len(nibbles)//2)])
b1_bytes2 = bytes([(nibbles[i*2+1] << 4) | nibbles[i*2] for i in range(len(nibbles)//2)])

print("B1 bytes 1:", b1_bytes1)
print("B1 bytes 2:", b1_bytes2)
