import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.uint8)

print("Total bytes:", len(raw))
print("Byte distribution min, max:", np.min(raw), np.max(raw))

# Check LSB (bit 0)
lsb = raw & 1
# Check if LSBs have ASCII or non-random structure
# Let's pack 8 LSBs into bytes
lsb_bytes = np.packbits(lsb)
print("LSB bytes len:", len(lsb_bytes))

# Search for flag formats in lsb_bytes: flag{, CTF{, etc.
# In both MSB-first and LSB-first bit order
for bitorder in ['big', 'little']:
    b = np.packbits(lsb, bitorder=bitorder).tobytes()
    for needle in [b'flag', b'FLAG', b'ctf', b'CTF', b'{', b'http']:
        if needle in b:
            pos = b.find(needle)
            print(f"Found {needle} in LSB (bitorder={bitorder}) at offset {pos}: {b[pos:pos+60]}")

# Also check bit 1, bit 2
for bit in range(1, 4):
    bits = (raw >> bit) & 1
    for bitorder in ['big', 'little']:
        b = np.packbits(bits, bitorder=bitorder).tobytes()
        for needle in [b'flag', b'FLAG', b'ctf', b'CTF']:
            if needle in b:
                pos = b.find(needle)
                print(f"Found {needle} in bit {bit} (bitorder={bitorder}) at offset {pos}: {b[pos:pos+60]}")
