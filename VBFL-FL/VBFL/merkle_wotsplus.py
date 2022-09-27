"""
Binary trees are represented by arrays.
The children of node #i are nodes #((i << 1) + 1) and #((i << 1) + 2)
The parent of #i is #((i - 1) >> 1)
"""

# change this line into
# import lamport as OTS
# to use lamport OTS instead
# import winternitz1 as OTS
import random

# import winternitz_wotsplus as OTS
from utils import shake128
from tqdm_alt import tqdm

import winternitz.signatures

class MerkleTree():

    def __init__(self, HEIGHT=10, progressbar=True):
        self.n_keys = 1 << HEIGHT
        r = range(self.n_keys)
        if progressbar:
            r = tqdm(r)
        keys = []
        wotslist = [winternitz.signatures.WOTSPLUS(seed=random.seed()) for _ in r]

        for wots in wotslist:
            keys.append((wots.privkey, wots.pubkey))
        tree = [None] * (self.n_keys - 1) + \
            [shake128(b''.join(k[1])) for k in keys]

        for i in reversed(range(self.n_keys - 1)):
            tree[i] = shake128(tree[(i << 1) + 1] + tree[(i << 1) + 2])

        self.keys = keys
        self.tree = tree
        self.wotskeys = wotslist
        # print(self.tree[0])
        self.last_key_used = -1

    @property
    def public_key(self):
        return self.tree[0]

    #Dev Added Code
    def check_key_index(self, key_used):
        if key_used+1 in range(self.n_keys):
            # print("Key_Index in Range -- DEV")
            return True
        else:
            # print("Key Index NOT In Rage  --- DEV")
            return False


    def signature(self, msg):
        msg = str(msg).encode("utf-8")
        key_index = self.last_key_used + 1
        assert key_index in range(self.n_keys), 'All keys have been used'

        secret, public = self.keys[key_index]
        wots = self.wotskeys[key_index]
        self.last_key_used = key_index

        auth = []
        for i, b in self.iter_ancestors(key_index + self.n_keys - 1):
            auth.append(self.tree[(i << 1) + 2 - b])

        auth = tuple(auth)
        wots_signature = wots.sign(msg)
        # print("1. WOTS signature TYPE................", type(wots_signature))
        # print("2. Key Index Signing==============================", key_index)


        # return (key_index, OTS.signature(msg, secret), public, auth)

        return (key_index, wots_signature, public, auth)

    @staticmethod
    def iter_ancestors(i):
        while i:
            i, b = divmod(i - 1, 2)
            yield i, b

    def verify(self, msg, otssig, merkle_public):
        msg = str(msg).encode("utf-8")
        key_index, sig, otspublic, auth = otssig
        # --------------------------------------
        # https://pypi.org/project/winternitz/
        # Another person or machine wants to verify your signature:
        # get required hash function by comparing the name
        # published with local implementaitons
        if sig["hashalgo"] == "openssl_sha512":
            # print("Hash Openssl_sha512")
            hashfunc = winternitz.signatures.openssl_sha512
        elif sig["hashalgo"] == "openssl_sha256":
            # print("Hash Openssl_sha256")
            hashfunc = winternitz.signatures.openssl_sha256
        else:
            # print("ERRORR")
            raise NotImplementedError("Hash function not implemented")
        # pubKey = winternitz.signatures.WOTSPLUS().getPubkeyFromSignature(message=msg, signature=sig)
        wots_other = winternitz.signatures.WOTSPLUS(w=sig["w"], hashfunction=hashfunc,
                                                    digestsize=sig["digestsize"], pubkey=sig["pubkey"],
                                                    seed=sig["seed"], prf=sig["prf"])
        # wots_other = winternitz.signatures.WOTSPLUS(w=sig["w"], hashfunction=hashfunc,
        #                                         digestsize=sig["digestsize"], pubkey=sig["pubkey"], seed=sig["seed"])
        success = wots_other.verify(message=msg, signature=sig["signature"])
        # print("SUCCEEEEEEEEEEESSSSSSSSS------------------", success)
        # ---------------------------------------

        h = shake128(b''.join(otspublic))
        n_keys = 1 << len(auth)

        # loop invariant: h is the value of node #i
        for a, (i, b) in zip(auth, MerkleTree.iter_ancestors(key_index + n_keys - 1)):
            h = shake128((a + h) if b else (h + a))
        # print("Merkle Public Valid?",h==merkle_public)

        # if not wots_other.verify(message=msg, signature=sig["signature"]):
        #     print("WOTS Verificatioin SUCCESS?", "FALSE")
        #     # print("WOTS Verification Going on ...................................")
        #     return False

        return h == merkle_public
