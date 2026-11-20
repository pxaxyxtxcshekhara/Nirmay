# Test LFSR whitening on all candidates

candidates = {
    'H1': b'\xef\xec\xbcc\x1e\x96L\x8f\xe1_8\xa1\xf2\xf3V\xc2\x07`0\x15\xf2M\xc7\xc3\x88\xc8I\x8d\xaf\xc3\x0f\xd4\xe3f\xb0\x10',
    'H2': b'\xfe\xce\xcb6\xe1i\xc4\xf8\x1e\xf5\x83\x1a/?e,p\x06\x03Q/\xd4|<\x88\x8c\x94\xd8\xfa<\xf0M>f\x0b\x01',
    'L1': b'\x1f\x1cLc\xe1\x96\xbc\x7f\x1e_7\xae\xfd\xf3V\xcd\x08`0\xe5\xfd\xb2\xc8\xc3w\xc7\xb9r\xaf\xc3\x0f+\x13f@\xe0',
    'L2': b"\xf1\xc1\xc46\x1ei\xcb\xf7\xe1\xf5s\xea\xdf?e\xdc\x80\x06\x03^\xdf+\x8c<w|\x9b'\xfa<\xf0\xb21f\x04\x0e"
}

def generate_lfsr_bytes(poly, init_state, n_bytes, shift_dir='right', output_bit='lsb', bit_order='msb_first'):
    # poly: e.g. 0x1D or 0xB8 or 0xE0 etc.
    state = init_state
    out = []
    for _ in range(n_bytes):
        byte_val = 0
        for b_idx in range(8):
            if shift_dir == 'right':
                # bit out is LSB
                bit = state & 1
                # feedback
                # taps from poly
                fb = 0
                temp = state & poly
                while temp:
                    fb ^= (temp & 1)
                    temp >>= 1
                state = (state >> 1) | (fb << 7)
            else: # left
                bit = (state >> 7) & 1
                fb = 0
                temp = state & poly
                while temp:
                    fb ^= (temp & 1)
                    temp >>= 1
                state = ((state << 1) & 0xFF) | fb
            
            if bit_order == 'msb_first':
                byte_val = (byte_val << 1) | bit
            else:
                byte_val |= (bit << b_idx)
        out.append(byte_val)
    return bytes(out)

# Polynomials for x^8 + x^6 + x^5 + x^4 + 1:
# Taps can be:
# If x^8 is feedback: x^6, x^5, x^4, x^0:
# In hex:
# 1 + x^4 + x^5 + x^6 + x^8 -> 0x71 or 0x8E or 0x1D or 0xB8
polys = [0x1D, 0xB8, 0x71, 0x8E, 0x2D, 0xB4, 0xD4, 0x2B]

found = []
for cname, cbytes in candidates.items():
    for poly in polys:
        for init in [0xFF, 0x01, 0xAA, 0x55]:
            for s_dir in ['right', 'left']:
                for b_ord in ['msb_first', 'lsb_first']:
                    keystream = generate_lfsr_bytes(poly, init, len(cbytes), s_dir, bit_order=b_ord)
                    dec = bytes([cbytes[i] ^ keystream[i] for i in range(len(cbytes))])
                    # Check printable ASCII ratio
                    printable = sum(32 <= b <= 126 for b in dec)
                    if printable >= len(dec) - 5 or b'flag' in dec.lower() or b'ctf' in dec.lower():
                        found.append((printable, cname, hex(poly), hex(init), s_dir, b_ord, dec))

found.sort(key=lambda x: x[0], reverse=True)
print(f"Top results (found {len(found)}):")
for f in found[:10]:
    print(f"Printable: {f[0]}/{len(candidates['H1'])}, Cand: {f[1]}, Poly: {f[2]}, Init: {f[3]}, Dir: {f[4]}, Ord: {f[5]}")
    print(f"  Decoded: {f[6]}")
