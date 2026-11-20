# Decode header of Burst 1

b1_header_bins = [39, 55, 127, 63, 119, 111, 95, 55]
b0 = 2

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

# For SF7 header:
# In gr-lora, header is ALWAYS reduced rate!
# "reduced_rate ? d_sf - 2u : d_sf"
# So ppm for header is SF - 2 = 7 - 2 = 5!
# And symbols are:
bin_idxs = [((b - b0) % 128) for b in b1_header_bins]

# Wait, in reduced_rate:
# std::lround(bin_idx / 4.0f) % d_number_of_bins_hdr!
# In SF7, header has reduced rate: bin / 4!
# Let's check:
for red_rate in [True, False]:
    if red_rate:
        b_in = [((b - b0) % 128) // 4 for b in b1_header_bins]
        ppm = 5
    else:
        b_in = [((b - b0) % 128) for b in b1_header_bins]
        ppm = 7
    
    words_gr = [x ^ (x >> 1) for x in b_in]
    deint = deinterleave(words_gr, ppm)
    deshuf = deshuffle(deint)
    print(f"RedRate={red_rate}, ppm={ppm}:")
    print(f"  deint:  {[hex(x) for x in deint]}")
    print(f"  deshuf: {[hex(x) for x in deshuf]}")
