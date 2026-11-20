import numpy as np

# Let's inspect the extracted symbols from Sym 0 to 71
symbols = [
    2014, 21, 64, 1971, 1951, 6, 2038, 2001, 1982, 107, 1990, 1974,
    55, 44, 2007, 84, 2012, 106, 1998, 1992, 52, 109, 89, 1979,
    2006, 1965, 2004, 1942, 82, 86, 11, 2010, 1917, 59, 14, 1994,
    36, 2032, 47, 32, 77, 1910, 53, 62, 1945, 143, 67, 63,
    76, 21, 1960, 1984, 112, 1978, 2038, 1958, 1947, 1988, 29, 2035,
    1994, 115, 78, 43, 117, 52, 96, 1922, 1914, 48, 72, 120
]

print(f"Number of symbols: {len(symbols)}")
print("Values:", symbols[:10])

# Check mod 4 (i.e. lowest 2 bits)
print("sym % 4:", [s % 4 for s in symbols[:20]])

# Wait! Notice how many symbols are near 2048 or near 0!
# In LoRa, dechirping can be inverted or shifted, or let's look at the distribution:
# 2014 = -34 mod 2048
# 1971 = -77 mod 2048
# 1951 = -97 mod 2048
# 2038 = -10 mod 2048
# 2001 = -47 mod 2048
# 1982 = -66 mod 2048
# Look! They are all small numbers either positive (21, 64, 6, 55, 44, 84...)
# or negative mod 2048 (-34, -77, -97, -10, -47, -66...)!
# Why are ALL values within [-150, +150] of 0?!
