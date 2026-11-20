# Test 9-bit LFSR x^9 + x^5 + 1

candidates = {
    'H1': b'\xef\xec\xbcc\x1e\x96L\x8f\xe1_8\xa1\xf2\xf3V\xc2\x07`0\x15\xf2M\xc7\xc3\x88\xc8I\x8d\xaf\xc3\x0f\xd4\xe3f\xb0\x10',
    'H2': b'\xfe\xce\xcb6\xe1i\xc4\xf8\x1e\xf5\x83\x1a/?e,p\x06\x03Q/\xd4|<\x88\x8c\x94\xd8\xfa<\xf0M>f\x0b\x01',
    'L1': b'\x1f\x1cLc\xe1\x96\xbc\x7f\x1e_7\xae\xfd\xf3V\xcd\x08`0\xe5\xfd\xb2\xc8\xc3w\xc7\xb9r\xaf\xc3\x0f+\x13f@\xe0',
    'L2': b"\xf1\xc1\xc46\x1ei\xcb\xf7\xe1\xf5s\xea\xdf?e\xdc\x80\x06\x03^\xdf+\x8c<w|\x9b'\xfa<\xf0\xb21f\x04\x0e"
}

def lfsr9(n_bytes, seed=0x1FF, shift_dir='right', bit_order='msb_first'):
    # x^9 + x^5 + 1
    # If shift_dir == 'right':
    # bits 8..0
    # taps: bit 0 and bit 4 (or 5)
    out = []
    state = seed
    for _ in range(n_bytes):
        byte_val = 0
        for b_idx in range(8):
            if shift_dir == 'right':
                bit = state & 1
                # feedback = bit 0 ^ bit 4 (or bit 5)
                fb = (state & 1) ^ ((state >> 4) & 1)
                state = (state >> 1) | (fb << 8)
            else: # left
                bit = (state >> 8) & 1
                fb = ((state >> 8) & 1) ^ ((state >> 4) & 1)
                state = ((state << 1) & 0x1FF) | fb
            
            if bit_order == 'msb_first':
                byte_val = (byte_val << 1) | bit
            else:
                byte_val |= (bit << b_idx)
        out.append(byte_val)
    return bytes(out)

for cname, cdata in candidates.items():
    for fb_tap in [4, 5]:
        for s_dir in ['right', 'left']:
            for b_ord in ['msb_first', 'lsb_first']:
                # simulate
                state = 0x1FF
                out = []
                for _ in range(len(cdata)):
                    bval = 0
                    for b_idx in range(8):
                        if s_dir == 'right':
                            bit = state & 1
                            fb = (state & 1) ^ ((state >> fb_tap) & 1)
                            state = (state >> 1) | (fb << 8)
                        else:
                            bit = (state >> 8) & 1
                            fb = ((state >> 8) & 1) ^ ((state >> fb_tap) & 1)
                            state = ((state << 1) & 0x1FF) | fb
                        if b_ord == 'msb_first':
                            bval = (bval << 1) | bit
                        else:
                            bval |= (bit << b_idx)
                    out.append(bval)
                ks = bytes(out)
                dec = bytes([cdata[i] ^ ks[i] for i in range(len(cdata))])
                if b'flag' in dec.lower() or b'ctf' in dec.lower() or sum(32 <= b <= 126 for b in dec) > 25:
                    print(f"Match: {cname}, tap={fb_tap}, dir={s_dir}, ord={b_ord}: {dec}")
