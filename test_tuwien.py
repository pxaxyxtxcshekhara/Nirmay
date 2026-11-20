def lora_tuwien_whitening(n_bytes):
    reg = 0xFF
    seq = []
    for _ in range(n_bytes):
        seq.append(reg)
        # feedback = mod(sum(dec2binvec(bitand(register, 0xB8))), 2)
        # 0xB8 = 0b10111000 -> taps at bits 7, 5, 4, 3 (or depending on 0-indexing)
        # bin(reg & 0xB8).count('1') % 2
        fb = bin(reg & 0xB8).count('1') % 2
        reg = ((reg * 2) & 0xFF) + fb
    return seq

seq = lora_tuwien_whitening(36)
print("TU Wien whitening sequence:", [hex(x) for x in seq])

candidates = {
    'H1': b'\xef\xec\xbcc\x1e\x96L\x8f\xe1_8\xa1\xf2\xf3V\xc2\x07`0\x15\xf2M\xc7\xc3\x88\xc8I\x8d\xaf\xc3\x0f\xd4\xe3f\xb0\x10',
    'H2': b'\xfe\xce\xcb6\xe1i\xc4\xf8\x1e\xf5\x83\x1a/?e,p\x06\x03Q/\xd4|<\x88\x8c\x94\xd8\xfa<\xf0M>f\x0b\x01',
    'L1': b'\x1f\x1cLc\xe1\x96\xbc\x7f\x1e_7\xae\xfd\xf3V\xcd\x08`0\xe5\xfd\xb2\xc8\xc3w\xc7\xb9r\xaf\xc3\x0f+\x13f@\xe0',
    'L2': b"\xf1\xc1\xc46\x1ei\xcb\xf7\xe1\xf5s\xea\xdf?e\xdc\x80\x06\x03^\xdf+\x8c<w|\x9b'\xfa<\xf0\xb21f\x04\x0e"
}

for name, cand in candidates.items():
    dec = bytes([b ^ w for b, w in zip(cand, seq)])
    print(f"{name}: {dec}")
