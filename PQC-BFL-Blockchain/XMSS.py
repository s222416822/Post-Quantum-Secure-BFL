# from WOTS import *
from typing import List
# from DataStructure import *

import sys

# from utils import *

from random import choice, seed, randint
from string import ascii_letters, digits
from hashlib import sha256
from math import floor, log2, log, ceil
# from DataStructure import ADRS

class XMSSPrivateKey:

    def __init__(self):
        self.wots_private_keys = None
        self.idx = None
        self.SK_PRF = None
        self.root_value = None
        self.SEED = None

class XMSSPublicKey:

    def __init__(self):
        self.OID = None
        self.root_value = None
        self.SEED = None

class XMSSKeypair:

    def __init__(self, SK, PK):
        self.SK = SK
        self.PK = PK

class SigXMSS:
    def __init__(self, idx_sig, r, sig, SK, M2):
        self.idx_sig = idx_sig
        self.r = r
        self.sig = sig
        self.SK = SK
        self.M2 = M2

class SigWithAuthPath:
    def __init__(self, sig_ots, auth):
        self.sig_ots = sig_ots
        self.auth = auth

class ADRS:

    def __init__(self):
        self.layerAddress = bytes(4)
        self.treeAddress = bytes(8)
        self.type = bytes(4)

        self.first_word = bytes(4)
        self.second_word = bytes(4)
        self.third_word = bytes(4)

        self.keyAndMask = bytes(4)

    def setType(self, type_value):
        self.type = type_value.to_bytes(4, byteorder='big')
        self.first_word = bytearray(4)
        self.second_word = bytearray(4)
        self.third_word = bytearray(4)
        self.keyAndMask = bytearray(4)

    def getTreeHeight(self):
        return self.second_word

    def getTreeIndex(self):
        return self.third_word

    def setHashAddress(self, value):
        self.third_word = value.to_bytes(4, byteorder='big')

    def setKeyAndMask(self, value):
        self.keyAndMask = value.to_bytes(4, byteorder='big')

    def setChainAddress(self, value):
        self.second_word = value.to_bytes(4, byteorder='big')

    def setTreeHeight(self, value):
        self.second_word = value.to_bytes(4, byteorder='big')

    def setTreeIndex(self, value):
        self.third_word = value.to_bytes(4, byteorder='big')

    def setOTSAddress(self, value):
        self.first_word = value.to_bytes(4, byteorder='big')

    def setLTreeAddress(self, value):
        self.first_word = value.to_bytes(4, byteorder='big')

    def setLayerAddress(self, value):
        self.layerAddress = value.to_bytes(4, byteorder='big')

    def setTreeAddress(self, value):
        self.treeAddress = value.to_bytes(4, byteorder='big')

def base_w(byte_string: bytes, w: int in {4, 16}, out_len):
    in_ = 0
    total_ = 0
    bits_ = 0
    base_w_ = []

    for i in range(0, out_len):
        if bits_ == 0:
            total_ = byte_string[in_]
            in_ += 1
            bits_ += 8

        bits_ -= log2(w)
        base_w_.append((total_ >> int(bits_)) & (w - 1))
    return base_w_


def generate_random_value(n):
    alphabet = ascii_letters + digits
    value = ''.join(choice(alphabet) for _ in range(n))
    return value


def compute_needed_bytes(n):
    if n == 0:
        return 1
    return int(log(n, 256)) + 1


def compute_lengths(n: int, w: int in {4, 16}):
    len_1 = ceil(8 * n / log2(w))
    len_2 = floor(log2(len_1 * (w - 1)) / log2(w)) + 1
    len_all = len_1 + len_2
    return len_1, len_2, len_all


def to_byte(value, bytes_count):
    return value.to_bytes(bytes_count, byteorder='big')


def xor(one: bytearray, two: bytearray) -> bytearray:
    return bytearray(a ^ b for (a, b) in zip(one, two))


def int_to_bytes(val, count):
    byteVal = to_byte(val, count)
    acc = bytearray()
    for i in range(len(byteVal)):
        if byteVal[i] < 16:
            acc.extend(b'0')
        curr = hex(byteVal[i])[2:]
        acc.extend(curr.encode())
    return acc


def F(KEY, M):
    key_len = len(KEY)
    toBytes = to_byte(0, 4)
    help_ = sha256(toBytes + KEY + M).hexdigest()[:key_len]
    out = bytearray()
    out.extend(map(ord, help_))
    return out


def chain(X, i, s, SEED, address, w):

    if s == 0:
        return X
    if (i + s) > (w - 1):
        return None
    tmp = chain(X, i, s - 1, SEED, address, w)

    address.setHashAddress((i + s - 1))
    address.setKeyAndMask(0)
    KEY = PRF(SEED, address)
    address.setKeyAndMask(1)
    BM = PRF(SEED, address)
    tmp = F(KEY, xor(tmp, BM))
    return tmp


def PRF(KEY: str, M: ADRS) -> bytearray:
    toBytes = to_byte(3, 4)
    key_len = len(KEY)
    KEY2 = bytearray()
    KEY2.extend(map(ord, KEY))
    help_ = sha256(toBytes + KEY2 + M.keyAndMask).hexdigest()[:key_len*2]
    out = bytearray()
    out.extend(map(ord, help_))
    return out


def H(KEY: bytearray, M: bytearray) -> bytearray:
    key_len = len(KEY)
    toBytes = to_byte(1, 4)
    help_ = sha256(toBytes + KEY + M).hexdigest()[:key_len]
    out = bytearray()
    out.extend(map(ord, help_))
    return out


def PRF_XMSS(KEY: str, M: bytearray, n: int) -> bytearray:
    toBytes = to_byte(3, 4)
    KEY2 = bytearray()
    KEY2.extend(map(ord, KEY))
    help_ = sha256(toBytes + KEY2 + M).hexdigest()[:n]
    out = bytearray()
    out.extend(map(ord, help_))
    return out


def H_msg(KEY: bytearray, M: bytearray, n: int) -> bytearray:
    toBytes = to_byte(2, 4)
    help_ = sha256(toBytes + KEY + M).hexdigest()[:n]
    out = bytearray()
    out.extend(map(ord, help_))
    return out


def RAND_HASH(left: bytearray, right: bytearray, SEED: str, adrs: ADRS):
    adrs.setKeyAndMask(0)
    KEY = PRF(SEED, adrs)
    adrs.setKeyAndMask(1)
    BM_0 = PRF(SEED, adrs)
    adrs.setKeyAndMask(2)
    BM_1 = PRF(SEED, adrs)

    return H(KEY, xor(left, BM_0) + xor(right, BM_1))


def pseudorandom_function(SEED, n):
    seed(SEED)
    sk_element = list()
    for i in range(n):
        sign = randint(0, 255)
        sk_element.append('{:02x}'.format(sign))

    return bytearray(''.join(sk_element).encode(encoding='utf-8'))


def WOTS_genSK(length, n):
    secret_key = [bytes()] * length

    for i in range(length):
        SEED = generate_random_value(length)

        secret_key[i] = pseudorandom_function(SEED, n)

    return secret_key


def WOTS_genPK(private_key: [bytes], length: int, w: int in {4, 16}, SEED, address):
    public_key = [bytes()] * length
    for i in range(length):
        address.setChainAddress(i)
        public_key[i] = chain(private_key[i], 0, w - 1, SEED, address, w)

    return public_key


def WOTS_sign(message: bytes, private_key: [bytes], w: int in {4, 16}, SEED, address):
    checksum = 0

    n = len(message) // 2
    len_1, len_2, len_all = compute_lengths(n, w)

    msg = base_w(message, w, len_1)

    for i in range(0, len_1):
        checksum += w - 1 - msg[i]

    checksum = checksum << int(8 - ((len_2 * log2(w)) % 8))

    len_2_bytes = compute_needed_bytes(checksum)

    msg.extend(base_w(to_byte(checksum, len_2_bytes), w, len_2))

    signature = [bytes()] * len_all
    print(signature)

    for i in range(0, len_all):
        address.setChainAddress(i)
        signature[i] = chain(private_key[i], 0, msg[i], SEED, address, w)

    return signature


def WOTS_pkFromSig(message: bytes, signature: [bytes], w: int in {4, 16}, address, SEED):
    checksum = 0

    n = len(message) // 2
    len_1, len_2, len_all = compute_lengths(n, w)

    msg = base_w(message, w, len_1)

    for i in range(0, len_1):
        checksum += w - 1 - msg[i]

    checksum = checksum << int(8 - ((len_2 * log2(w)) % 8))

    len_2_bytes = compute_needed_bytes(checksum)

    msg.extend(base_w(to_byte(checksum, len_2_bytes), w, len_2))

    tmp_pk = [bytes()] * len_all

    for i in range(0, len_all):
        address.setChainAddress(i)
        tmp_pk[i] = chain(signature[i], msg[i], w - 1 - msg[i], SEED, address, w)

    return tmp_pk


def ltree(pk: List[bytearray], address: ADRS, SEED: str, length: int) -> bytearray:

    address.setTreeHeight(0)

    while length > 1:
        for i in range(floor(length / 2)):
            address.setTreeIndex(i)
            pk[i] = RAND_HASH(pk[2 * i], pk[2 * i + 1], SEED, address)

        if length % 2 == 1:
            pk[floor(length / 2)] = pk[length - 1]

        length = ceil(length / 2)
        height = address.getTreeHeight()
        height = int.from_bytes(height, byteorder='big')
        address.setTreeHeight(height + 1)

    return pk[0]


def treeHash(SK: XMSSPrivateKey, s: int, t: int, address: ADRS, w: int in {4, 16}, length_all: int) -> bytearray:

    class StackElement:
        def __init__(self, node_value=None, height=None):
            self.node_value = node_value
            self.height = height

    Stack = []

    if s % (1 << t) != 0:
        raise ValueError("should be s % (1 << t) != 0")

    for i in range(0, int(pow(2, t))):
        SEED = SK.SEED
        address.setType(0)
        address.setOTSAddress(s + i)
        pk = WOTS_genPK(SK.wots_private_keys[s + i], length_all, w, SEED, address)
        address.setType(1)
        address.setLTreeAddress(s + i)
        node = ltree(pk, address, SEED, length_all)

        node_as_stack_element = StackElement(node, 0)

        address.setType(2)
        address.setTreeHeight(0)
        address.setTreeIndex(i + s)

        while len(Stack) != 0 and Stack[len(Stack) - 1].height == node_as_stack_element.height:
            address.setTreeIndex(int((int.from_bytes(address.getTreeHeight(), byteorder='big') - 1) / 2))

            previous_height = node_as_stack_element.height

            node = RAND_HASH(Stack.pop().node_value, node_as_stack_element.node_value, SEED, address)

            node_as_stack_element = StackElement(node, previous_height + 1)

            address.setTreeHeight(int.from_bytes(address.getTreeHeight(), byteorder='big') + 1)

        Stack.append(node_as_stack_element)

    return Stack.pop().node_value


def XMSS_keyGen(height: int, n: int, w: int in {4, 16}) -> XMSSKeypair:

    len_1, len_2, len_all = compute_lengths(n, w)

    wots_sk = []
    for i in range(0, 2 ** height):
        wots_sk.append(WOTS_genSK(len_all, n))

    SK = XMSSPrivateKey()
    PK = XMSSPublicKey()
    idx = 0

    SK.SK_PRF = generate_random_value(n)
    SEED = generate_random_value(n)
    SK.SEED = SEED
    SK.wots_private_keys = wots_sk

    adrs = ADRS()

    root = treeHash(SK, 0, height, adrs, w, len_all)

    SK.idx = idx
    SK.root_value = root

    PK.OID = generate_random_value(n)
    PK.root_value = root
    PK.SEED = SEED

    KeyPair = XMSSKeypair(SK, PK)
    return KeyPair


def buildAuth(SK: XMSSPrivateKey, index: int, address: ADRS, w: int in {4, 16}, length_all: int, h: int) -> List[bytearray]:
    auth = []

    for j in range(h):
        k = floor(index / (2 ** j)) ^ 1
        auth.append(treeHash(SK, k * (2 ** j), j, address, w, length_all))
    return auth


def treeSig(message: bytearray, SK: XMSSPrivateKey, address: ADRS, w: int in {4, 16}, length_all: int, idx_sig: int, h: int) -> SigWithAuthPath:
    auth = buildAuth(SK, idx_sig, address, w, length_all, h)
    address.setType(0)
    address.setOTSAddress(idx_sig)
    sig_ots = WOTS_sign(message, SK.wots_private_keys[idx_sig], w, SK.SEED, address)
    Sig = SigWithAuthPath(sig_ots, auth)
    return Sig


def XMSS_sign(message: bytearray, SK: XMSSPrivateKey, w: int in {4, 16}, address: ADRS, h: int) -> SigXMSS:
    n = len(message) // 2
    len_1, len_2, length_all = compute_lengths(n, w)
    idx_sig = SK.idx
    SK.idx = idx_sig + 1
    r = PRF_XMSS(SK.SK_PRF, to_byte(idx_sig, 4), len_1)
    arrayOfBytes = bytearray()
    arrayOfBytes.extend(r)
    arrayOfBytes.extend(SK.root_value)
    arrayOfBytes.extend(bytearray(int_to_bytes(idx_sig, n)))
    M2 = H_msg(arrayOfBytes, message, len_1)

    value = treeSig(M2, SK, address, w, length_all, idx_sig, h)

    return SigXMSS(idx_sig, r, value, SK, M2)


def XMSS_rootFromSig(idx_sig: int, sig_ots, auth: List[bytearray], message: bytearray, h: int, w: int in {4, 16}, SEED, address: ADRS):
    n = len(message) // 2
    len_1, len_2, length_all = compute_lengths(n, w)

    address.setType(0)
    address.setOTSAddress(idx_sig)
    pk_ots = WOTS_pkFromSig(message, sig_ots, w, address, SEED)
    address.setType(1)
    address.setLTreeAddress(idx_sig)
    node = [bytearray, bytearray]
    node[0] = ltree(pk_ots, address, SEED, length_all)
    address.setType(2)
    address.setTreeIndex(idx_sig)

    for k in range(0, h):
        address.setTreeHeight(k)
        if floor(idx_sig / (2 ** k)) % 2 == 0:
            address.setTreeIndex(int.from_bytes(address.getTreeIndex(), byteorder='big') // 2)
            node[1] = RAND_HASH(node[0], auth[k], SEED, address)
        else:
            address.setTreeIndex((int.from_bytes(address.getTreeIndex(), byteorder='big') - 1) // 2)
            node[1] = RAND_HASH(auth[k], node[0], SEED, address)

        node[0] = node[1]

    return node[0]


def XMSS_verify(Sig: SigXMSS, M: bytearray, PK: XMSSPublicKey, w: int in {4, 16}, SEED, height: int):

    address = ADRS()

    n = len(M) // 2
    len_1, len_2, length_all = compute_lengths(n, w)

    arrayOfBytes = bytearray()
    arrayOfBytes.extend(Sig.r)
    arrayOfBytes.extend(PK.root_value)
    arrayOfBytes.extend(bytearray(int_to_bytes(Sig.idx_sig, n)))

    M2 = H_msg(arrayOfBytes, M, len_1)

    node = XMSS_rootFromSig(Sig.idx_sig, Sig.sig.sig_ots, Sig.sig.auth, M2, height, w, SEED, address)

    if node == PK.root_value:
        return True
    else:
        return False

