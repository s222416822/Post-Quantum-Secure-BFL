# reference - https://developer.ibm.com/technologies/blockchain/tutorials/develop-a-blockchain-application-from-scratch-in-python/
import copy
import json
from hashlib import sha256


class Block:
    def __init__(self, idx, previous_block_hash=None, transactions=None, nonce=0,
                 miner_xmss_pub_key=None,
                 mined_by=None, mining_rewards=None, pow_proof=None, signature=None, tree=None, miner_tree_no=None):
        self._idx = idx
        self._previous_block_hash = previous_block_hash
        self._transactions = transactions
        self._nonce = nonce
        self._miner_xmss_pub_key = miner_xmss_pub_key
        self._mined_by = mined_by
        self._mining_rewards = mining_rewards
        self._pow_proof = pow_proof
        self._signature = signature
        self._tree = tree
        self._tree_no = miner_tree_no

    def compute_hash(self, hash_entire_block=False):
        block_content = copy.deepcopy(self.__dict__)
        if not hash_entire_block:
            block_content['_pow_proof'] = None
            block_content['_signature'] = None
            block_content['_mining_rewards'] = None
        return sha256(str(sorted(block_content.items())).encode('utf-8')).hexdigest()

    def remove_signature_for_verification(self):
        self._signature = None

    def remove_tree_for_verification(self):
        self._tree = None

    def set_pow_proof(self, the_hash):
        self._pow_proof = the_hash

    def nonce_increment(self):
        self._nonce += 1

    def return_previous_block_hash(self):
        return self._previous_block_hash

    def return_block_idx(self):
        return self._idx

    def return_pow_proof(self):
        return self._pow_proof

    def return_miner_xmss_pub_key(self):
        return self._miner_xmss_pub_key

    def set_previous_block_hash(self, hash_to_set):
        self._previous_block_hash = hash_to_set

    def add_verified_transaction(self, transaction):
        self._transactions.append(transaction)

    def set_nonce(self, nonce):
        self._nonce = nonce

    def set_mined_by(self, mined_by):
        self._mined_by = mined_by

    def return_mined_by(self):
        return self._mined_by

    def set_tree(self, tree):
        self._tree = tree

    def set_tree_no(self, tree_no):
        self._tree_no = tree_no

    def return_tree_no(self):
        self._tree_no

    def return_tree(self):
        return self._tree

    def set_signature(self, signature):
        self._signature = signature

    def return_signature(self):
        return self._signature

    def set_mining_rewards(self, mining_rewards):
        self._mining_rewards = mining_rewards

    def return_mining_rewards(self):
        return self._mining_rewards

    def return_transactions(self):
        return self._transactions

    def free_tx(self):
        try:
            del self._transactions
        except:
            pass
