from typing import List
from random import choice, seed, randint
from string import ascii_letters, digits
from hashlib import sha256
from math import floor, log2, log, ceil


class XMSSPrivateKey:
    # XMSS Private Key structure
    def __init__(self):
        self.wots_private_keys = None  # WOTS private keys
        self.idx = None  # Index for signing
        self.SK_PRF = None  # Secret PRF key
        self.root_value = None  # Root value of the tree
        self.SEED = None  # Seed for PRF


class XMSSPublicKey:
    # XMSS Public Key structure
    def __init__(self):
        self.OID = None  # Object Identifier
        self.root_value = None  # Root value of the tree
        self.SEED = None  # Seed for public key operations


class XMSSKeypair:
    # Keypair structure containing private and public keys
    def __init__(self, SK, PK):
        self.SK = SK  # Private key
        self.PK = PK  # Public key


class SigXMSS:
    # XMSS Signature structure
    def __init__(self, idx_sig, r, sig, SK, M2):
        self.idx_sig = idx_sig  # Signature index
        self.r = r  # Random value
        self.sig = sig  # Signature value
        self.SK = SK  # Private key
        self.M2 = M2  # Message hash


class SigWithAuthPath:
    # Signature with authentication path structure
    def __init__(self, sig_ots, auth):
        self.sig_ots = sig_ots  # OTS signature
        self.auth = auth  # Authentication path


class ADRS:
    # Address structure for tree-based operations
    def __init__(self):
        self.layerAddress = bytes(4)  # Layer address
        self.treeAddress = bytes(8)  # Tree address
        self.type = bytes(4)  # Address type

        self.first_word = bytes(4)
        self.second_word = bytes(4)
        self.third_word = bytes(4)

        self.keyAndMask = bytes(4)  # Key and mask value

    def setType(self, type_value):
        # Set address type value
        self.type = type_value.to_bytes(4, byteorder='big')
        self.first_word = bytearray(4)
        self.second_word = bytearray(4)
        self.third_word = bytearray(4)
        self.keyAndMask = bytearray(4)

    def getTreeHeight(self):
        # Get tree height
        return self.second_word

    def getTreeIndex(self):
        # Get tree index
        return self.third_word

    def setHashAddress(self, value):
        # Set hash address
        self.third_word = value.to_bytes(4, byteorder='big')

    def setKeyAndMask(self, value):
        # Set key and mask value
        self.keyAndMask = value.to_bytes(4, byteorder='big')

    def setChainAddress(self, value):
        # Set chain address
        self.second_word = value.to_bytes(4, byteorder='big')

    def setTreeHeight(self, value):
        # Set tree height value
        self.second_word = value.to_bytes(4, byteorder='big')

    def setTreeIndex(self, value):
        # Set tree index value
        self.third_word = value.to_bytes(4, byteorder='big')

    def setOTSAddress(self, value):
        # Set OTS address value
        self.first_word = value.to_bytes(4, byteorder='big')

    def setLTreeAddress(self, value):
        # Set L-tree address value
        self.first_word = value.to_bytes(4, byteorder='big')

    def setLayerAddress(self, value):
        # Set layer address value
        self.layerAddress = value.to_bytes(4, byteorder='big')

    def setTreeAddress(self, value):
        # Set tree address value
        self.treeAddress = value.to_bytes(8, byteorder='big')


# Helper function: base_w converts byte string into w-based output
def base_w(byte_string: bytes, w: int, out_len):
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


# Generate a random value of length n
def generate_random_value(n):
    alphabet = ascii_letters + digits
    value = ''.join(choice(alphabet) for _ in range(n))
    return value


# Compute needed bytes to represent value n
def compute_needed_bytes(n):
    if n == 0:
        return 1
    return int(log(n, 256)) + 1


# Compute length values for Winternitz OTS
def compute_lengths(n: int, w: int):
    len_1 = ceil(8 * n / log2(w))
    len_2 = floor(log2(len_1 * (w - 1)) / log2(w)) + 1
    len_all = len_1 + len_2
    return len_1, len_2, len_all


# Convert integer to bytes
def to_byte(value, bytes_count):
    return value.to_bytes(bytes_count, byteorder='big')


# XOR operation for two bytearrays
def xor(one: bytearray, two: bytearray) -> bytearray:
    return bytearray(a ^ b for (a, b) in zip(one, two))


# Convert integer to hexadecimal byte array
def int_to_bytes(val, count):
    byteVal = to_byte(val, count)
    acc = bytearray()
    for i in range(len(byteVal)):
        if byteVal[i] < 16:
            acc.extend(b'0')
        curr = hex(byteVal[i])[2:]
        acc.extend(curr.encode())
    return acc


# Hash function F using SHA-256
def F(KEY, M):
    key_len = len(KEY)
    toBytes = to_byte(0, 4)
    help_ = sha256(toBytes + KEY + M).hexdigest()[:key_len]
    out = bytearray()
    out.extend(map(ord, help_))
    return out


# Chain function for WOTS signature generation
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


# Pseudorandom function (PRF) using SHA-256
def PRF(KEY: str, M: ADRS) -> bytearray:
    toBytes = to_byte(3, 4)
    key_len = len(KEY)
    KEY2 = bytearray()
    KEY2.extend(map(ord, KEY))
    help_ = sha256(toBytes + KEY2 + M.keyAndMask).hexdigest()[:key_len * 2]
    out = bytearray()
    out.extend(map(ord, help_))
    return out


# Hash function H using SHA-256
def H(KEY: bytearray, M: bytearray) -> bytearray:
    key_len = len(KEY)
    toBytes = to_byte(1, 4)
    help_ = sha256(toBytes + KEY + M).hexdigest()[:key_len]
    out = bytearray()
    out.extend(map(ord, help_))
    return out


# Pseudorandom function for XMSS (PRF_XMSS)
def PRF_XMSS(KEY: str, M: bytearray, n: int) -> bytearray:
    toBytes = to_byte(3, 4)
    KEY2 = bytearray()
    KEY2.extend(map(ord, KEY))
    help_ = sha256(toBytes + KEY2 + M).hexdigest()[:n]
    out = bytearray()
    out.extend(map(ord, help_))
    return out


# Hash function H_msg for XMSS
def H_msg(KEY: bytearray, M: bytearray, n: int) -> bytearray:
    toBytes = to_byte(2, 4)
    help_ = sha256(toBytes + KEY + M).hexdigest()[:n]
    out = bytearray()
    out.extend(map(ord, help_))
    return out


# Randomized hash for tree operations
def RAND_HASH(left: bytearray, right: bytearray, SEED: str, adrs: ADRS):
    adrs.setKeyAndMask(0)
    KEY = PRF(SEED, adrs)
    adrs.setKeyAndMask(1)
    BM_0 = PRF(SEED, adrs)
    adrs.setKeyAndMask(2)
    BM_1 = PRF(SEED, adrs)

    return H(KEY, xor(left, BM_0) + xor(right, BM_1))


def pseudorandom_function(SEED, n):
    # Generate a pseudorandom value of n bytes using the provided SEED
    seed(SEED)
    sk_element = list()
    for i in range(n):
        sign = randint(0, 255)
        sk_element.append('{:02x}'.format(sign))

    return bytearray(''.join(sk_element).encode(encoding='utf-8'))


def WOTS_genSK(length, n):
    # Generate a WOTS private key of a given length and size n
    secret_key = [bytes()] * length

    for i in range(length):
        SEED = generate_random_value(length)
        secret_key[i] = pseudorandom_function(SEED, n)

    return secret_key


def WOTS_genPK(private_key: [bytes], length: int, w: int, SEED, address):
    # Generate a WOTS public key from the private key using chaining
    public_key = [bytes()] * length
    for i in range(length):
        address.setChainAddress(i)
        public_key[i] = chain(private_key[i], 0, w - 1, SEED, address, w)

    return public_key


def WOTS_sign(message: bytes, private_key: [bytes], w: int, SEED, address):
    # Create WOTS signature for a message
    checksum = 0
    n = len(message) // 2
    len_1, len_2, len_all = compute_lengths(n, w)

    msg = base_w(message, w, len_1)

    # Calculate checksum
    for i in range(0, len_1):
        checksum += w - 1 - msg[i]

    checksum = checksum << int(8 - ((len_2 * log2(w)) % 8))
    len_2_bytes = compute_needed_bytes(checksum)

    # Extend the message with the checksum
    msg.extend(base_w(to_byte(checksum, len_2_bytes), w, len_2))

    # Generate signature by chaining private key values
    signature = [bytes()] * len_all
    for i in range(0, len_all):
        address.setChainAddress(i)
        signature[i] = chain(private_key[i], 0, msg[i], SEED, address, w)

    return signature


def WOTS_pkFromSig(message: bytes, signature: [bytes], w: int, address, SEED):
    # Derive WOTS public key from signature and message
    checksum = 0
    n = len(message) // 2
    len_1, len_2, len_all = compute_lengths(n, w)

    msg = base_w(message, w, len_1)

    # Calculate checksum
    for i in range(0, len_1):
        checksum += w - 1 - msg[i]

    checksum = checksum << int(8 - ((len_2 * log2(w)) % 8))
    len_2_bytes = compute_needed_bytes(checksum)

    # Extend the message with the checksum
    msg.extend(base_w(to_byte(checksum, len_2_bytes), w, len_2))

    # Recreate public key by chaining signature values
    tmp_pk = [bytes()] * len_all
    for i in range(0, len_all):
        address.setChainAddress(i)
        tmp_pk[i] = chain(signature[i], msg[i], w - 1 - msg[i], SEED, address, w)

    return tmp_pk


def ltree(pk: List[bytearray], address: ADRS, SEED: str, length: int) -> bytearray:
    # Compute L-tree from WOTS public key
    address.setTreeHeight(0)

    # Combine pairs of nodes into one until only the root remains
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


def treeHash(SK: XMSSPrivateKey, s: int, t: int, address: ADRS, w: int, length_all: int) -> bytearray:
    # Perform tree hash from index s over 2^t leaves

    class StackElement:
        # Define stack element to hold node values and heights
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

        # Generate WOTS public key and calculate L-tree root
        pk = WOTS_genPK(SK.wots_private_keys[s + i], length_all, w, SEED, address)
        address.setType(1)
        address.setLTreeAddress(s + i)
        node = ltree(pk, address, SEED, length_all)

        node_as_stack_element = StackElement(node, 0)
        address.setType(2)
        address.setTreeHeight(0)
        address.setTreeIndex(i + s)

        # Combine nodes in the stack to build the tree
        while len(Stack) != 0 and Stack[len(Stack) - 1].height == node_as_stack_element.height:
            address.setTreeIndex(int((int.from_bytes(address.getTreeHeight(), byteorder='big') - 1) / 2))
            previous_height = node_as_stack_element.height

            # Perform hash on node pairs
            node = RAND_HASH(Stack.pop().node_value, node_as_stack_element.node_value, SEED, address)
            node_as_stack_element = StackElement(node, previous_height + 1)
            address.setTreeHeight(int.from_bytes(address.getTreeHeight(), byteorder='big') + 1)

        Stack.append(node_as_stack_element)

    return Stack.pop().node_value


def XMSS_keyGen(height: int, n: int, w: int) -> XMSSKeypair:
    # Generate keypair (XMSSPrivateKey and XMSSPublicKey)

    len_1, len_2, len_all = compute_lengths(n, w)

    # Generate WOTS private keys for all leaf nodes
    wots_sk = []
    for i in range(0, 2 ** height):
        wots_sk.append(WOTS_genSK(len_all, n))

    SK = XMSSPrivateKey()  # XMSS private key
    PK = XMSSPublicKey()  # XMSS public key
    idx = 0

    # Set random PRF seed and SEED for SK
    SK.SK_PRF = generate_random_value(n)
    SEED = generate_random_value(n)
    SK.SEED = SEED
    SK.wots_private_keys = wots_sk

    adrs = ADRS()  # Initialize address

    # Generate root node of the tree
    root = treeHash(SK, 0, height, adrs, w, len_all)

    # Set root and index for SK and PK
    SK.idx = idx
    SK.root_value = root

    PK.OID = generate_random_value(n)
    PK.root_value = root
    PK.SEED = SEED

    # Return keypair
    KeyPair = XMSSKeypair(SK, PK)
    return KeyPair


def buildAuth(SK: XMSSPrivateKey, index: int, address: ADRS, w: int, length_all: int, h: int) -> List[bytearray]:
    # Build the authentication path

    auth = []
    for j in range(h):
        k = floor(index / (2 ** j)) ^ 1
        # Append hash of sibling node in the tree
        auth.append(treeHash(SK, k * (2 ** j), j, address, w, length_all))
    return auth


def treeSig(message: bytearray, SK: XMSSPrivateKey, address: ADRS, w: int, length_all: int, idx_sig: int,
            h: int) -> SigWithAuthPath:
    # Generate tree-based signature

    # Build authentication path
    auth = buildAuth(SK, idx_sig, address, w, length_all, h)
    address.setType(0)
    address.setOTSAddress(idx_sig)

    # Generate WOTS signature for message
    sig_ots = WOTS_sign(message, SK.wots_private_keys[idx_sig], w, SK.SEED, address)

    # Return signature and authentication path
    Sig = SigWithAuthPath(sig_ots, auth)
    return Sig


def XMSS_sign(message: bytearray, SK: XMSSPrivateKey, w: int, address: ADRS, h: int) -> SigXMSS:
    # XMSS signing process

    n = len(message) // 2
    len_1, len_2, length_all = compute_lengths(n, w)
    idx_sig = SK.idx
    SK.idx = idx_sig + 1

    # Generate random value for r
    r = PRF_XMSS(SK.SK_PRF, to_byte(idx_sig, 4), len_1)

    # Create message hash M2
    arrayOfBytes = bytearray()
    arrayOfBytes.extend(r)
    arrayOfBytes.extend(SK.root_value)
    arrayOfBytes.extend(bytearray(int_to_bytes(idx_sig, n)))
    M2 = H_msg(arrayOfBytes, message, len_1)

    # Generate signature
    value = treeSig(M2, SK, address, w, length_all, idx_sig, h)

    # Return XMSS signature object
    return SigXMSS(idx_sig, r, value, SK, M2)


def XMSS_rootFromSig(idx_sig: int, sig_ots, auth: List[bytearray], message: bytearray, h: int, w: int, SEED,
                     address: ADRS):
    # Reconstruct the root node from signature and authentication path

    n = len(message) // 2
    len_1, len_2, length_all = compute_lengths(n, w)

    address.setType(0)
    address.setOTSAddress(idx_sig)

    # Get WOTS public key from signature
    pk_ots = WOTS_pkFromSig(message, sig_ots, w, address, SEED)
    address.setType(1)
    address.setLTreeAddress(idx_sig)

    # Compute L-tree and get node
    node = [bytearray, bytearray]
    node[0] = ltree(pk_ots, address, SEED, length_all)
    address.setType(2)
    address.setTreeIndex(idx_sig)

    # Traverse tree and reconstruct root node
    for k in range(0, h):
        address.setTreeHeight(k)
        if floor(idx_sig / (2 ** k)) % 2 == 0:
            address.setTreeIndex(int.from_bytes(address.getTreeIndex(), byteorder='big') // 2)
            node[1] = RAND_HASH(node[0], auth[k], SEED, address)
        else:
            address.setTreeIndex((int.from_bytes(address.getTreeIndex(), byteorder='big') - 1) // 2)
            node[1] = RAND_HASH(auth[k], node[0], SEED, address)

        node[0] = node[1]

    # Return reconstructed root
    return node[0]


def XMSS_verify(Sig: SigXMSS, M: bytearray, PK: XMSSPublicKey, w: int, SEED, height: int):
    # Verify XMSS signature

    address = ADRS()

    n = len(M) // 2
    len_1, len_2, length_all = compute_lengths(n, w)

    # Recreate message hash M2
    arrayOfBytes = bytearray()
    arrayOfBytes.extend(Sig.r)
    arrayOfBytes.extend(PK.root_value)
    arrayOfBytes.extend(bytearray(int_to_bytes(Sig.idx_sig, n)))

    M2 = H_msg(arrayOfBytes, M, len_1)

    # Reconstruct root node from signature
    node = XMSS_rootFromSig(Sig.idx_sig, Sig.sig.sig_ots, Sig.sig.auth, M2, height, w, SEED, address)

    # Verify if the reconstructed root matches the public key's root
    if node == PK.root_value:
        return True
    else:
        return False
