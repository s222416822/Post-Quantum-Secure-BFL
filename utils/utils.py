from os import urandom
from keccak import SHAKE128, SHAKE128_ITER


# Generate cryptographically secure random bytes of length l
def getrandom(l):
    return bytes(urandom(l))


# Convert message bytes into bits, yielding True for '1' and False for '0'
def bits(msg):
    for by in msg:
        for bi in format(by, '08b'):
            yield bi == '1'


# Split each byte into two 4-bit chunks (nibbles)
def chunks4(msg):
    for by in msg:
        yield from divmod(by, 16)


# Perform SHAKE128 hashing with an output length of 32 bytes
def shake128(m):
    return SHAKE128(m, 32)


# Perform iterative SHAKE128 hashing with an iterator
def shake128_iter(it, m):
    return SHAKE128_ITER(it, m, 32)
