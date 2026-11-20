import numpy as np

# Let's test the header decoding on the first 8 symbols
# Candidate bins:
raw_bins = [1792, 184, 528, 1444, 1288, 60, 1980, 1684]

def rotl(val, shift, bits):
    return ((val << shift) | (val >> (bits - shift))) & ((1 << bits) - 1)

def rotr(val, shift, bits):
    return ((val >> shift) | (val << (bits - shift))) & ((1 << bits) - 1)

def deinterleave(words, ppm, rotate_left=True):
    bits_per_word = len(words)
    offset_start = ppm - 1
    words_deinterleaved = [0] * ppm
    for i in range(bits_per_word):
        if rotate_left:
            word = rotl(words[i], i, ppm)
        else:
            word = rotr(words[i], i, ppm)
        for x in range(ppm):
            bit = (word >> x) & 1
            words_deinterleaved[x] |= (bit << i)
    return words_deinterleaved

shuffle_pattern = [5, 0, 1, 2, 4, 3, 6, 7]

def deshuffle(words):
    res = []
    for w in words:
        r = 0
        for j, s in enumerate(shuffle_pattern):
            r |= ((w >> s) & 1) << j
        res.append(r)
    return res

# Let's test across possible b0 offsets around 16
for b0 in range(12, 20):
    bin_idxs = [((b - b0) % 2048) // 4 for b in raw_bins]
    
    # Test Gray:
    # Option A: word = bin_idx ^ (bin_idx >> 1)
    words_A = [x ^ (x >> 1) for x in bin_idxs]
    # Option B: standard Gray decode
    def gray_dec(g):
        b = g
        while g > 0:
            g >>= 1
            b ^= g
        return b
    words_B = [gray_dec(x) for x in bin_idxs]
    # Option C: raw bin_idx
    words_C = list(bin_idxs)
    
    for opt_name, words in [('A', words_A), ('B', words_B), ('C', words_C)]:
        for rot_l in [True, False]:
            deint = deinterleave(words, 9, rotate_left=rot_l)
            deshuf = deshuffle(deint)
            # The header is 5 nibbles (or 5 codewords):
            # Nibble 0: payload length high nibble
            # Nibble 1: payload length low nibble
            # Nibble 2: CR (3 bits), CRC flag (1 bit)
            # Nibble 3, 4: header checksum
            # Let's inspect the nibbles (lowest 4 bits):
            nibbles = [w & 0xF for w in deshuf]
            # Payload length: (nibble 0 << 4) | nibble 1, or (nibble 1 << 4) | nibble 0
            len1 = (nibbles[0] << 4) | nibbles[1]
            len2 = (nibbles[1] << 4) | nibbles[0]
            # Also without deshuffle:
            nibbles_raw = [w & 0xF for w in deint]
            len_raw1 = (nibbles_raw[0] << 4) | nibbles_raw[1]
            
            # Print if payload length is reasonable, e.g. between 10 and 60
            for name, n_arr, l_val in [('deshuf', nibbles, len1), ('raw', nibbles_raw, len_raw1)]:
                if 10 <= l_val <= 60:
                    print(f"b0={b0}, opt={opt_name}, rot_l={rot_l}, {name}: len={l_val}, nibbles={n_arr[:5]}")
