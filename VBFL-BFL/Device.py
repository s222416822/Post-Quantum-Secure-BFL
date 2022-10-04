import sys
from struct import unpack

import numpy as np
import torch
from matplotlib import cm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from datetime import datetime

import oqs
from DatasetLoad import DatasetLoad
from DatasetLoad import AddGaussianNoise
from torch import optim
import random
import copy
import time
from sys import getsizeof
# https://cryptobook.nakov.com/digital-signatures/rsa-sign-verify-examples
from Crypto.PublicKey import RSA
from hashlib import sha256
from Models import Mnist_2NN, Mnist_CNN
from Blockchain import Blockchain
import vrf_helpers
import winternitz.signatures as wotsplus
import pandas as pd
from merkle_wotsplus import MerkleTree
import winternitz.signatures
from utils import shake128

from XMSS import *

import matplotlib.pyplot as plt

alg = {"Dilithium5", "Falcon-1024"}

dict = {
    "device_1":{"public_key":0, "xmss_pub_keys":[], "signatures":[]},
    "device_2":{"public_key":0, "xmss_pub_keys":[], "signatures":[]},
    "device_3":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_4":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_5":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_6":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_7":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_8":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_9":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_10":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_11":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_12":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_13":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_14":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_15":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_16":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_17":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_18":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_19":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
    "device_20":{"public_key":0,  "xmss_pub_keys":[],"signatures":[]},
}

devices_dict = {

}
file_malicious = open("check_malicious.txt", "a")



benchmarkDict = {
    "xPkSize": [],
    "xSigSize":[],
    "xSigSizeStr":[],
    "dPkSize":[],
    "dSigSize":[],
    "xGen":[],
    "xSign":[],
    "xVerify":[],
    "dGen":[],
    "dSign":[],
    "dVerify":[],

}

import os

if not os.path.exists('benchmark_d'):
    os.makedirs('benchmark_d')

date_time = datetime.now().strftime("%m%d%Y_%H%M%S")

# 1. parse arguments and save to file
# create folder of logs
bench_folder = f"benchmark_d"

benchmarkFile = open(f"{bench_folder}/benchmark.txt", "a")

class Device:
    def __init__(self, idx, assigned_train_ds, assigned_test_dl, local_batch_size, learning_rate, loss_func, opti, network_stability, net, dev, miner_acception_wait_time, miner_accepted_transactions_size_limit, validator_threshold, pow_difficulty, even_link_speed_strength, base_data_transmission_speed, even_computation_power, is_malicious, noise_variance, check_signature, not_resync_chain, malicious_updates_discount, knock_out_rounds, lazy_worker_knock_out_rounds):
        self.idx = idx
        # deep learning variables
        self.train_ds = assigned_train_ds
        self.test_dl = assigned_test_dl
        self.local_batch_size = local_batch_size
        self.loss_func = loss_func
        self.network_stability = network_stability
        self.net = copy.deepcopy(net)
        if opti == "SGD":
            self.opti = optim.SGD(self.net.parameters(), lr=learning_rate)
        self.dev = dev
        # in real system, new data can come in, so train_dl should get reassigned before training when that happens
        self.train_dl = DataLoader(self.train_ds, batch_size=self.local_batch_size, shuffle=True)
        self.local_train_parameters = None
        self.initial_net_parameters = None
        self.global_parameters = None
        # blockchain variables
        self.role = None
        self.pow_difficulty = pow_difficulty
        if even_link_speed_strength:
            self.link_speed = base_data_transmission_speed
        else:
            self.link_speed = random.random() * base_data_transmission_speed
        self.devices_dict = None
        self.aio = False
        ''' simulating hardware equipment strength, such as good processors and RAM capacity. Following recorded times will be shrunk by this value of times
        # for workers, its update time
        # for miners, its PoW time
        # for validators, its validation time
        # might be able to simulate molopoly on computation power when there are block size limit, as faster devices' transactions will be accepted and verified first
        '''
        # if even_computation_power:
        # 	self.computation_power = 1
        # else:
        # self.computation_power = random.randint(1, 3)
        self.computation_power = 0
        self.peer_list = set()
        # used in cross_verification and in the PoS
        self.online = True
        self.rewards = 0
        self.blockchain = Blockchain()
        # init key pair
        # self.modulus = None
        # self.private_key = None
        # self.public_key = None
        # self.generate_rsa_key()

        self.dilithium_private_key = None
        self.dilithium_public_key = None
        self.dilithium_signatures = []  # to store dilithium signatures


        self.generate_dilithium_key()

        # xmss key pair
        self.treeNo = 0
        self.xmss_tree = None
        self.xmss_public_key = None
        self.xmss_public_keys = []  # to store xmss public keys
        self.trees = []  # to store xmss trees
        self.generate_xmss_key()

        self.dictionary = {
            "signatures":[],
            # "xmss_pub_keys":[],
            "d_pub_key": None
        }

        # black_list stores device index rather than the object
        self.black_list = set()
        self.knock_out_rounds = knock_out_rounds
        self.lazy_worker_knock_out_rounds = lazy_worker_knock_out_rounds
        self.worker_accuracy_accross_records = {}
        self.has_added_block = False
        self.the_added_block = None
        self.is_malicious = is_malicious
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        self.not_resync_chain = not_resync_chain
        self.malicious_updates_discount = malicious_updates_discount
        # used to identify slow or lazy workers
        self.active_worker_record_by_round = {}
        self.untrustworthy_workers_record_by_comm_round = {}
        self.untrustworthy_validators_record_by_comm_round = {}
        # for picking PoS legitimate blockd;bs
        # self.stake_tracker = {} # used some tricks in main.py for ease of programming
        # used to determine the slowest device round end time to compare PoW with PoS round end time. If simulate under computation_power = 0, this may end up equaling infinity
        self.round_end_time = 0
        ''' For workers '''
        self.local_updates_rewards_per_transaction = 0
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0
        ''' For validators '''
        self.validator_associated_worker_set = set()
        self.validation_rewards_this_round = 0
        self.accuracies_this_round = {}
        self.validator_associated_miner = None
        # when validator directly accepts workers' updates
        self.unordered_arrival_time_accepted_worker_transactions = {}
        self.validator_accepted_broadcasted_worker_transactions = None or []
        self.final_transactions_queue_to_validate = {}
        self.post_validation_transactions_queue = None or []
        self.validator_threshold = validator_threshold
        self.validator_local_accuracy = None
        ''' For miners '''
        self.miner_associated_worker_set = set()
        self.miner_associated_validator_set = set()
        # dict cannot be added to set()
        self.unconfirmmed_transactions = None or []
        self.broadcasted_transactions = None or []
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        # when miner directly accepts validators' updates
        self.unordered_arrival_time_accepted_validator_transactions = {}
        self.miner_accepted_broadcasted_validator_transactions = None or []
        self.final_candidate_transactions_queue_to_mine = {}
        self.block_generation_time_point = None
        self.unordered_propagated_block_processing_queue = {} # pure simulation queue and does not exist in real distributed system
        ''' For malicious node '''
        self.variance_of_noises = None or []

        #DEV ADDED
        self.wotsplus = wotsplus.WOTSPLUS(seed=random.seed())
        self.stake_clip = 0
        self.selection_value = 0
        self.shape_value = 0
        self.vrf_output = 0
        self.contribution_value = 0
        self.bannedfor = 0
        self.detectedMalicious = False
        self.validation_loss = 0
        self.validation_accuracy = 0


    ''' Common Methods '''

    def decrease_banned_for(self):
        self.bannedfor = self.bannedfor -1

    ''' setters '''

    def set_computation_power(self, value):
        self.computation_power = value

    def set_banned_for(self, banned):
        self.bannedfor = banned

    def set_vrf_output(self, output):
        self.vrf_output = output

    def set_shape_value(self, value):
        self.shape_value = value

    def set_selection_value(self, value):
        self.selection_value = value

    def set_contribution_value(self, value):
        self.contribution_value = value

    def set_devices_dict_and_aio(self, devices_dict, aio):
        self.devices_dict = devices_dict
        self.aio = aio

    # def generate_rsa_key(self):
    # 	keyPair = RSA.generate(bits=1024)
    # 	self.modulus = keyPair.n
    # 	self.private_key = keyPair.d
    # 	self.public_key = keyPair.e
    #

    def generate_dilithium_key(self):
        # keypair = oqs.Signature("Dilithium5")
        t = time.time()
        keypair = oqs.Signature("Falcon-1024")
        benchmarkDict["dGen"].append(time.time() - t)
        self.dilithium_public_key = keypair.generate_keypair()
        self.dilithium_private_key = keypair.export_secret_key()
        dict[self.idx]["public_key"] = self.dilithium_public_key
        benchmarkDict["dPkSize"].append(sys.getsizeof(self.dilithium_public_key))
        # benchmarkDict["dSkSize"].append(sys.getsizeof(self.dilithium_private_key))

    def generate_xmss_key(self):
        self.treeNo += 1
        # keypair = XMSS_keyGen(4, 256, 16)

        tx = time.time()
        tree = MerkleTree(HEIGHT=6)
        benchmarkDict["xGen"].append(time.time() - tx)
        self.xmss_tree = tree
        self.xmss_public_key = self.xmss_tree.public_key
        benchmarkDict["xPkSize"].append(sys.getsizeof(self.xmss_public_key))
        # print("Before Signing by Dilithium",self.xmss_public_key)
        # self.xmss_public_keys.append(self.xmss_public_key)
        # print(type(self.xmss_public_key))
        # print(self.xmss_public_key)
        # self.dilithium_signatures.append(self.sign_dilithium_msg(self.xmss_public_key))
        # signer = oqs.Signature("Dilithium5", self.dilithium_private_key)
        signer = oqs.Signature("Falcon-1024", self.dilithium_private_key)
        # print("1.Hash of Dilithium Pub ky", sha256(str(self.dilithium_public_key).encode()).digest())

        # hash = int.from_bytes(sha256(str(self.xmss_public_key).encode('utf-8')).digest(), byteorder='big')
        # print("Dilithium Signing Hash", hash)
        dict[self.idx]["xmss_pub_keys"].append(self.xmss_public_key)
        t = time.time()
        signature = signer.sign(self.xmss_public_key)
        benchmarkDict["dSign"].append(time.time() - t)
        benchmarkDict["dSigSize"].append(sys.getsizeof(signature))

        # print(signature)
        # print("Signing ====  Signature Hash", sha256(str(signature).encode()).digest())
        # print("1TYPE", type(self.xmss_public_key))
        # print("2TYPE", type(self.dilithium_public_key))
        # print("3TYPE", type(signature))

        dict[self.idx]["signatures"].append(signature)
        self.dilithium_signatures.append(signature)


    # def generate_xmss_key(self):
    #     self.treeNo += 1
    #     tree = MerkleTree(HEIGHT=6)
    #     self.xmss_tree = tree
    #     self.xmss_public_key = self.xmss_tree.public_key
    #     # print("Before Signing by Dilithium",self.xmss_public_key)
    #     # self.xmss_public_keys.append(self.xmss_public_key)
    #     # print(type(self.xmss_public_key))
    #     # print(self.xmss_public_key)
    #     # self.dilithium_signatures.append(self.sign_dilithium_msg(self.xmss_public_key))
    #     # signer = oqs.Signature("Dilithium5", self.dilithium_private_key)
    #     signer = oqs.Signature("Falcon-1024", self.dilithium_private_key)
    #     # print("1.Hash of Dilithium Pub ky", sha256(str(self.dilithium_public_key).encode()).digest())
    #
    #     # hash = int.from_bytes(sha256(str(self.xmss_public_key).encode('utf-8')).digest(), byteorder='big')
    #     # print("Dilithium Signing Hash", hash)
    #     dict[self.idx]["xmss_pub_keys"].append(self.xmss_public_key)
    #     signature = signer.sign(self.xmss_public_key)
    #     # print(signature)
    #     # print("Signing ====  Signature Hash", sha256(str(signature).encode()).digest())
    #     # print("1TYPE", type(self.xmss_public_key))
    #     # print("2TYPE", type(self.dilithium_public_key))
    #     # print("3TYPE", type(signature))
    #
    #     dict[self.idx]["signatures"].append(signature)
    #     self.dilithium_signatures.append(signature)



    # print(pd.DataFrame.from_dict(dict))

    def init_global_parameters(self):
        self.initial_net_parameters = self.net.state_dict()
        self.global_parameters = self.net.state_dict()

    def vrf(self, alphastring):
        value = []
        secret_key = self.wotsplus.privkey
        # TODO Add SHA function to secret key of WOTS+ before sending to VRF Pk
        vrf_sk = str(secret_key).encode("utf-8")
        vrf_pk = vrf_helpers.get_public_key(vrf_sk)

        p_status, pi_string, signature = vrf_helpers.ecvrf_prove(vrf_sk, alphastring, self.wotsplus)

        verify = self.wotsplus.verify(pi_string, signature=signature["signature"])

        print("Verify VRF", verify)

        b_status, beta_string = vrf_helpers.ecvrf_proof_to_hash(pi_string)
        beta_sum = sum(list(beta_string))
        result, beta_string2 = vrf_helpers.ecvrf_verify(vrf_pk, pi_string, alphastring)
        if p_status == "VALID" and \
                b_status == "VALID" and \
                result == "VALID" and \
                beta_string == beta_string2:
            print("Commitment verified")
            return True, beta_string
        return False, b'null'

    def bytes_to_float(self, byte_value):
        return float(unpack('L', sha256(byte_value).digest()[:8])[0]) / 2 ** (len(byte_value))

    def assign_role(self, comm_round, stake, stake_average):
    # def assign_role(self):
        file_record = open(f"{bench_folder}/file_record.txt", "a")
        value_encoded = str(self.blockchain.return_last_block()).encode()
        value_encoded = "Hello".encode()
        result, beta_string = self.vrf(value_encoded)
        ratio = self.bytes_to_float(beta_string)
        msg1 = f"VRF Output: {ratio} | Stake: {stake}| Stake Average: {stake_average}"

        msg = ""
        # equal probability
        role_choice = random.randint(0, 2)
        print(role_choice)
        if role_choice == 0:
            self.role = "worker"
            # print("Worker")
            msg = f"COMM_ROUND: {comm_round}| {self.idx} with stake: {self.return_stake()} selected WORKER"
        elif role_choice == 1:
            self.role = "miner"
            # print("Miner")
            msg = f"COMM_ROUND: {comm_round}| {self.idx} with stake: {self.return_stake()} selected MINER"
        else:
            self.role = "validator"
            # print("Validator")
            msg = f"COMM_ROUND: {comm_round}| {self.idx} with stake: {self.return_stake()} selected VALIDATOR"
        if comm_round > 10:
            file_record.write(msg1 + "\n" + msg + "\n")
        return msg, msg1

    # used for hard_assign
    def assign_miner_role(self):
        self.role = "miner"

    def assign_worker_role(self):
        self.role = "worker"

    def assign_validator_role(self):
        self.role = "validator"

    ''' getters '''

    def return_banned_for(self):
        return self.bannedfor

    def return_contribution_value(self):
        return self.contribution_value

    def return_vrf_output(self):
        return self.vrf_output

    def return_shape_value(self):
        return self.shape_value

    def return_selection_value(self):
        return self.selection_value

    def return_idx(self):
        return self.idx

    # def return_rsa_pub_key(self):
    #     return {"modulus": self.modulus, "pub_key": self.public_key}

    def return_dilithium_signatures(self):
        return self.dilithium_signatures

    def return_dilithium_pub_key(self):
        return self.dilithium_public_key

    def return_dilithium_priv_key(self):
        return self.dilithium_private_key

    def return_xmss_pub_key(self):
        return {"pub_key": self.xmss_public_key}

    # def return_xmss_pub_key(self):
    #     return self.xmss_public_key

    # def return_tree_no(self):
    #     return {"tree_no": self.treeNo}

    def return_tree_no(self):
        return self.treeNo

    def return_peers(self):
        return self.peer_list

    def return_role(self):
        return self.role

    def is_online(self):
        return self.online

    def return_is_malicious(self):
        return self.is_malicious

    def return_black_list(self):
        return self.black_list

    def return_blockchain_object(self):
        return self.blockchain

    def return_stake(self):
        return self.rewards

    def return_computation_power(self):
        return self.computation_power

    def return_the_added_block(self):
        return self.the_added_block

    def return_round_end_time(self):
        return self.round_end_time

    ''' functions '''

    # def sign_msg(self, msg):
    #     hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
    #     # pow() is python built-in modular expone            if device.is_malicious and device.is_online():
    #                 device.set_contribution_value(-1)
    #                 # device.set_banned_for(3)   #ban for 3 comm rounds
    #             elif device.is_online() and not device.is_malicious:
    #                 device.set_contribution_value(1)
    #             elif not device.is_online():
    #                 device.set_contribution_value(0)            if device.is_malicious and device.is_online():
    #                 device.set_contribution_value(-1)
    #                 # device.set_banned_for(3)   #ban for 3 comm rounds
    #             elif device.is_online() and not device.is_malicious:
    #                 device.set_contribution_value(1)
    #             elif not device.is_online():
    #                 device.set_contribution_value(0)ntiation function
    #     signature = pow(hash, self.private_key, self.modulus)
    #     return signature

    # def sign_dilithium_msg(self, msg, device):
    #     signer = oqs.Signature("Dilithium5", self.dilithium_private_key)
    #     hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
    #     print(hash)
    #     # pow() is python built-in modular exponentiation function
    #     # signature = pow(hash, self.private_key, self.modulus)
    #     signature = signer.sign(hash)
    #     self.dilithium_signatures.append(signature)  #store dilithium signature
    #     return signature

    def verify_dilithium(self, msg, device_id, tree_no):

        ds = time.time()
        # peers_list = self.return_peers()
        # for peer in peers_list:
        #     if peer.return_idx == device_id:
        #         signature = peer.return_dilithium_signatures()
        #         public_key = peer.return_dilithium_pub_key()
        #         d_public_key = public_key
        #
        #         d_signature = signature[tree_no - 1]
        # hash = int.from_bytes(sha256(msg).digest(),
        #                       byteorder='big')
        # print("Verifying Dilithium", hash)
        # d_public_key = dict[device_id]["public_key"]
        d_public_key = self.devices_dict[device_id].return_dilithium_pub_key()

        d_signature = self.devices_dict[device_id].return_dilithium_signatures()[tree_no - 1]
        # access public key from directly peerslist

        # access signature directly from peerslist


        # print("Hash from Dict", dict[device_id]["xmss_pub_keys"][tree_no-1])
        # print("For verifying xmss pub key", msg)
        # print("2.Hash of Dilithium Pub ky", sha256(str(d_public_key).encode()).digest())
        # d_signature = dict[device_id]["signatures"][tree_no - 1]
        # print("Verifying ==== ", sha256(str(d_signature).encode()).digest())
        #
        # print("1TYPE", type(msg))
        # print("2TYPE", type(d_public_key))
        # print("3TYPE", type(d_signature))
        # print(d_signature)
        # result = oqs.Signature("Dilithium5").verify(msg, d_signature, d_public_key)
        result = oqs.Signature("Falcon-1024").verify(msg, d_signature, d_public_key)
        benchmarkDict["dVerify"].append(time.time() - ds)
        return result

    # def verify_dilithium(self, msg, device_id, tree_no):
    #     # hash = int.from_bytes(sha256(msg).digest(),
    #     #                       byteorder='big')
    #     # print("Verifying Dilithium", hash)
    #     d_public_key = dict[device_id]["public_key"]
    #     # print("Hash from Dict", dict[device_id]["xmss_pub_keys"][tree_no-1])
    #     # print("For verifying xmss pub key", msg)
    #     # print("2.Hash of Dilithium Pub ky", sha256(str(d_public_key).encode()).digest())
    #     d_signature = dict[device_id]["signatures"][tree_no - 1]
    #     # print("Verifying ==== ", sha256(str(d_signature).encode()).digest())
    #     #
    #     # print("1TYPE", type(msg))
    #     # print("2TYPE", type(d_public_key))
    #     # print("3TYPE", type(d_signature))
    #     # print(d_signature)
    #     # result = oqs.Signature("Dilithium5").verify(msg, d_signature, d_public_key)
    #     result = oqs.Signature("Falcon-1024").verify(msg, d_signature, d_public_key)
    #     return result

    def check_xmss_tree_index(self):
        if self.xmss_tree.check_key_index(self.xmss_tree.last_key_used):
            pass
        else:
            self.generate_xmss_key()

    def sign_msg_xmss(self, msg):
        # self.check_xmss_tree_index()
        # hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # hash = sha256(str(msg).encode('utf-8')).digest()
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # message = str(mg)

        # pow() is python built-in modular exponentiation function
        # signature = pow(hash, self.private_key, self.modulus)
        signingTime = time.time()
        signature = self.xmss_tree.signature(hash)
        signingTime = time.time() - signingTime
        benchmarkDict["xSign"].append(signingTime)
        benchmarkDict["xSigSize"].append(sys.getsizeof(signature))
        benchmarkDict["xSigSizeStr"].append(sys.getsizeof(str(signature)))

        # print("XMSS public key.....Signing", self.xmss_tree.public_key)
        # print("Signature Hash...Signing", sha256(str(signature).encode("utf-8")).digest())
        # print("3. Signed by==============================", self.idx, self.role)
        return signature

    def verify_msg_xmss(self, msg, otssig, merkle_public):

        xv = time.time()
        msg = str(msg).encode("utf-8")
        key_index, sig, otspublic, auth = otssig

        # print(sig["prf"])
        # print(sig["seed"])
        # if sig["hashalgo"] == "openssl_sha512":
        #     print("Hash Openssl_sha512")
        #     hashfunc = winternitz.signatures.openssl_sha512
        # elif sig["hashalgo"] == "openssl_sha256":
        #     print("Hash Openssl_sha256")
        #     hashfunc = winternitz.signatures.openssl_sha256
        # else:
        #     print("ERRORR")
        #     raise NotImplementedError("Hash function not implemented")
        # # pubKey = winternitz.signatures.WOTSPLUS().getPubkeyFromSignature(message=msg, signature=sig)
        # wots_other = winternitz.signatures.WOTSPLUS(w=sig["w"], hashfunction=hashfunc,
        #                                             digestsize=sig["digestsize"], pubkey=sig["pubkey"],
        #                                             seed=sig["seed"], prf=sig["prf"])
        # # wots_other = winternitz.signatures.WOTSPLUS(w=sig["w"], hashfunction=hashfunc,
        # #                                         digestsize=sig["digestsize"], pubkey=sig["pubkey"], seed=sig["seed"])
        # success = wots_other.verify(message=msg, signature=sig["signature"])

        h = shake128(b''.join(otspublic))
        n_keys = 1 << len(auth)

        # loop invariant: h is the value of node #i
        for a, (i, b) in zip(auth, MerkleTree.iter_ancestors(key_index + n_keys - 1)):
            h = shake128((a + h) if b else (h + a))
        # print("Merkle Public Valid?",h==merkle_public)


        wots_other = winternitz.signatures.WOTSPLUS(seed=sig["seed"], pubkey=otspublic)

        print("WOTS Plus Verification ---------------------------------", wots_other.verify(message=msg, signature=sig["signature"]))

        if not wots_other.verify(message=msg, signature=sig["signature"]):
            print("WOTS Verificatioin SUCCESS?", "FALSE")
            # print("WOTS Verification Going on ...................................")
            return False

        benchmarkDict["xVerify"].append(time.time() - xv)
        benchmarkFile.write(f"{benchmarkDict}\n")


        return h == merkle_public

    # return success

    def add_peers(self, new_peers):
        if isinstance(new_peers, Device):
            self.peer_list.add(new_peers)
        else:
            self.peer_list.update(new_peers)

    def remove_peers(self, peers_to_remove):
        if isinstance(peers_to_remove, Device):
            self.peer_list.discard(peers_to_remove)
        else:
            self.peer_list.difference_update(peers_to_remove)

    def online_switcher(self, log_files_folder_path, conn, conn_cursor, num_of_offline):
        print("Switching the Device Online or Offline!1!!!!!!!!1")

        old_status = self.online
        online_indicator = random.random()  #dev: Returns 0.0 and 1.0

        # contribution_value = self.return_contribution_value()
        # print("Contribution Value", contribution_value)
        # print("Self Returned Banned for", self.return_banned_for())

        #DEV make sure the device was not malicious in previous round and prevent it from being online

        if online_indicator < 0.8:
            # if not contribution_value == -1 and self.return_banned_for() <= 0:
            self.online = True
            if old_status == False:
                print(f"{self.idx} goes back online.")
                # update peer list
                self.update_peer_list()
                # wo# resync chain
                if self.pow_resync_chain():
                    self.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
            # else:
            #     self.online = False
            #     if self.return_banned_for() > 0:
            #         self.decrease_banned_for()  # make sure the banned is for 3 communication round from when it was identified as malicioius
            # if back online, update peer and resync chain

            # if online_indicator < self.network_stability:
        # if online_indicator < 0.8:
        #     if not contribution_value == -1 and self.return_banned_for() <= 0:
        #         self.online = True
        #         if old_status == False:
        #             print(f"{self.idx} goes back online.")
        #             # update peer list
        #             self.update_peer_list()
        #             # wo# resync chain
        #             if self.pow_resync_chain():
        #                 self.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)
        #     else:
        #         self.online = False
        #         if self.return_banned_for() > 0:
        #             self.decrease_banned_for()  # make sure the banned is for 3 communication round from when it was identified as malicioius
        #     # if back online, update peer and resync chain

        else:
            num_of_offline += 1
            self.online = False
            print(f"{self.idx} goes offline.")


        # return self.online
        return num_of_offline

    def online_switcher_before(self):

        old_status = self.online
        online_indicator = random.random()  #dev: Returns 0.0 and 1.0
        if online_indicator < 0.5:
            self.online = True
            # if back online, update peer and resync chain
            if old_status == False:
                print(f"{self.idx} goes back online.")
                # update peer list
                self.update_peer_list()
                # resync chain
                if self.pow_resync_chain():
                    self.update_model_after_chain_resync()
        else:
            self.online = False
            print(f"{self.idx} goes offline.")
        return self.online

    def update_peer_list(self):
        print(f"\n{self.idx} - {self.role} is updating peer list...")
        old_peer_list = copy.copy(self.peer_list)
        online_peers = set()
        for peer in self.peer_list:
            if peer.is_online():
                online_peers.add(peer)
        # for online peers, suck in their peer list
        for online_peer in online_peers:
            self.add_peers(online_peer.return_peers())
        # remove itself from the peer_list if there is
        self.remove_peers(self)
        # remove malicious peers
        potential_malicious_peer_set = set()
        for peer in self.peer_list:
            if peer.return_idx() in self.black_list:
                potential_malicious_peer_set.add(peer)
        self.remove_peers(potential_malicious_peer_set)
        # print updated peer result
        if old_peer_list == self.peer_list:
            print("Peer list NOT changed.")
        else:
            print("Peer list has been changed.")
            added_peers = self.peer_list.difference(old_peer_list)
            if potential_malicious_peer_set:
                print("These malicious peers are removed")
                # for peer in removed_peers:
                for peer in self.remove_peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            if added_peers:
                print("These peers are added")
                for peer in added_peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            print("Final peer list:")
            for peer in self.peer_list:
                print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
            print()
        # WILL ALWAYS RETURN TRUE AS OFFLINE PEERS WON'T BE REMOVED ANY MORE, UNLESS ALL PEERS ARE MALICIOUS...but then it should not register with any other peer. Original purpose - if peer_list ends up empty, randomly register with another device
        return False if not self.peer_list else True

    def check_pow_proof(self, block_to_check):
        # remove its block hash(compute_hash() by default) to verify pow_proof as block hash was set after pow
        pow_proof = block_to_check.return_pow_proof()
        # print("pow_proof", pow_proof)
        # print("compute_hash", block_to_check.compute_hash())
        return pow_proof.startswith('0' * self.pow_difficulty) and pow_proof == block_to_check.compute_hash()

    def check_chain_validity(self, chain_to_check):
        chain_len = chain_to_check.return_chain_length()
        if chain_len == 0 or chain_len == 1:
            pass
        else:
            chain_to_check = chain_to_check.return_chain_structure()
            for block in chain_to_check[1:]:
                if self.check_pow_proof(block) and block.return_previous_block_hash() == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_entire_block=True):
                    pass
                else:
                    return False
        return True

    def accumulate_chain_stake(self, chain_to_accumulate):
        accumulated_stake = 0
        chain_to_accumulate = chain_to_accumulate.return_chain_structure()
        for block in chain_to_accumulate:
            accumulated_stake += self.devices_dict[block.return_mined_by()].return_stake()
        return accumulated_stake

    def resync_chain(self, mining_consensus):
        if self.not_resync_chain:
            return # temporary workaround to save GPU memory
        if mining_consensus == 'PoW':
            self.pow_resync_chain()
        else:
            self.pos_resync_chain()

    def pos_resync_chain(self):
        print(f"{self.role} {self.idx} is looking for a chain with the highest accumulated miner's stake in the network...")
        highest_stake_chain = None
        updated_from_peer = None
        curr_chain_stake = self.accumulate_chain_stake(self.return_blockchain_object())
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                peer_chain_stake = self.accumulate_chain_stake(peer_chain)
                if peer_chain_stake > curr_chain_stake:
                    if self.check_chain_validity(peer_chain):
                        print(f"A chain from {peer.return_idx()} with total stake {peer_chain_stake} has been found (> currently compared chain stake {curr_chain_stake}) and verified.")
                        # Higher stake valid chain found!
                        curr_chain_stake = peer_chain_stake
                        highest_stake_chain = peer_chain
                        updated_from_peer = peer.return_idx()
                    else:
                        print(f"A chain from {peer.return_idx()} with higher stake has been found BUT NOT verified. Skipped this chain for syncing.")
        if highest_stake_chain:
            # compare chain difference
            highest_stake_chain_structure = highest_stake_chain.return_chain_structure()
            # need more efficient machenism which is to reverse updates by # of blocks
            self.return_blockchain_object().replace_chain(highest_stake_chain_structure)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")
            #return block_iter
            return True
        print("Chain not resynced.")
        return False

    def pow_resync_chain(self):
        print(f"{self.role} {self.idx} is looking for a longer chain in the network...")
        longest_chain = None
        updated_from_peer = None
        curr_chain_len = self.return_blockchain_object().return_chain_length()
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                if peer_chain.return_chain_length() > curr_chain_len:
                    if self.check_chain_validity(peer_chain):
                        print(f"A longer chain from {peer.return_idx()} with chain length {peer_chain.return_chain_length()} has been found (> currently compared chain length {curr_chain_len}) and verified.")
                        # Longer valid chain found!
                        curr_chain_len = peer_chain.return_chain_length()
                        longest_chain = peer_chain
                        updated_from_peer = peer.return_idx()
                    else:
                        print(f"A longer chain from {peer.return_idx()} has been found BUT NOT verified. Skipped this chain for syncing.")
        if longest_chain:
            # compare chain difference
            longest_chain_structure = longest_chain.return_chain_structure()
            # need more efficient machenism which is to reverse updates by # of blocks
            self.return_blockchain_object().replace_chain(longest_chain_structure)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")
            #return block_iter
            return True
        print("Chain not resynced.")
        return False

    def set_stake_clip(self, stake_clip):
        self.stake_clip = stake_clip

    def return_stake_clip(self):
        self.stake_clip

    def receive_rewards(self, rewards):
        self.rewards += rewards

    # def verify_miner_transaction_by_signature(self, transaction_to_verify, miner_device_idx):
    # 	if miner_device_idx in self.black_list:
    # 		print(f"{miner_device_idx} is in miner's blacklist. Trasaction won't get verified.")
    # 		return False
    # 	if self.check_signature:
    # 		transaction_before_signed = copy.deepcopy(transaction_to_verify)
    # 		del transaction_before_signed["miner_signature"]
    # 		modulus = transaction_to_verify['miner_rsa_pub_key']["modulus"]
    # 		pub_key = transaction_to_verify['miner_rsa_pub_key']["pub_key"]
    # 		signature = transaction_to_verify["miner_signature"]
    # 		# verify
    # 		hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
    # 		hashFromSignature = pow(signature, pub_key, modulus)
    # 		if hash == hashFromSignature:
    # 			print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
    # 			return True
    # 		else:
    # 			print(f"Signature invalid. Transaction recorded by {miner_device_idx} is NOT verified.")
    # 			return False
    # 	else:
    # 		print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
    # 		return True
    def verify_miner_transaction_by_signature(self, transaction_to_verify, miner_device_idx):
        if miner_device_idx in self.black_list:
            print(f"{miner_device_idx} is in miner's blacklist. Trasaction won't get verified.")
            return False
        if self.check_signature:
            transaction_before_signed = copy.deepcopy(transaction_to_verify)
            del transaction_before_signed["miner_signature"]
            # modulus = transaction_to_verify['miner_rsa_pub_key']["modulus"]
            # pub_key = transaction_to_verify['miner_rsa_pub_key']["pub_key"]
            pub_key = transaction_to_verify['miner_xmss_pub_key']["pub_key"]
            signature = transaction_to_verify["miner_signature"]
            miner_tree_no = transaction_to_verify["miner_tree_no"]
            # xmss_tree = transaction_to_verify["xmss_tree_miner"]

            # hash = sorted(transaction_before_signed.items())
            hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(),
                                  byteorder='big')
            # verify
            # hash = int.from_bytes(sha256(str()).encode('utf-8')).digest(), byteorder='big')
            # hashFromSignature = pow(signature, pub_key, modulus)
            # if hash == hashFromSignature:
            if self.verify_msg_xmss(hash, signature, pub_key) and self.verify_dilithium(pub_key, miner_device_idx,
                                                                                        miner_tree_no):
                print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
                return True
            else:
                print(f"Signature invalid. Transaction recorded by {miner_device_idx} is NOT verified.")
                return False
        else:
            print(f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
            return True

    def verify_block(self, block_to_verify, sending_miner):
        # if not self.online_switcher():
        if not self.is_online():
            print(f"{self.idx} goes offline when verifying a block")
            return False, False
        verification_time = time.time()
        mined_by = block_to_verify.return_mined_by()
        if sending_miner in self.black_list:
            print(f"The miner propagating/sending this block {sending_miner} is in {self.idx}'s black list. Block will not be verified.")
            return False, False
        if mined_by in self.black_list:
            print(f"The miner {mined_by} mined this block is in {self.idx}'s black list. Block will not be verified.")
            return False, False
        # check if the proof is valid(verify _block_hash).
        if not self.check_pow_proof(block_to_verify):
            print(f"PoW proof of the block from miner {self.idx} is not verified.")
            return False, False
        # # check if miner's signature is valid
        if self.check_signature:
            signature_dict = block_to_verify.return_miner_xmss_pub_key()
            # modulus = signature_dict["modulus"]
            pub_key = signature_dict["pub_key"]
            signature = block_to_verify.return_signature()
            tree_no = block_to_verify.return_tree_no()

            # verify signature
            block_to_verify_before_sign = copy.deepcopy(block_to_verify)
            block_to_verify_before_sign.remove_signature_for_verification()

            # hash = block_to_verify_before_sign.__dict__
            hash = int.from_bytes(sha256(str(block_to_verify_before_sign.__dict__).encode('utf-8')).digest(),
                                  byteorder='big')
            # hash = int.from_bytes(sha256(str(block_to_verify_before_sign.__dict__).encode('utf-8')).digest(), byteorder='big')
            # hashFromSignature = pow(signature, pub_key, modulus)
            # if hash != hashFromSignature:
            if not self.verify_msg_xmss(hash, signature, pub_key) and not self.verify_dilithium(pub_key, mined_by, tree_no):
                #
                print(
                    f"Signature of the block sent by miner {sending_miner} mined by miner {mined_by} is not verified by {self.role} {self.idx}.")
                return False, False
            # check previous hash based on own chain
            last_block = self.return_blockchain_object().return_last_block()
            if last_block is not None:
                # check if the previous_hash referred in the block and the hash of latest block in the chain match.
                last_block_hash = last_block.compute_hash(hash_entire_block=True)
                if block_to_verify.return_previous_block_hash() != last_block_hash:
                    print(f"Block sent by miner {sending_miner} mined by miner {mined_by} has the previous hash recorded as {block_to_verify.return_previous_block_hash()}, but the last block's hash in chain is {last_block_hash}. This is possibly due to a forking event from last round. Block not verified and won't be added. Device needs to resync chain next round.")
                    return False, False
        # All verifications done.
        print(f"Block accepted from miner {sending_miner} mined by {mined_by} has been verified by {self.idx}!")
        verification_time = (time.time() - verification_time)/self.computation_power
        return block_to_verify, verification_time

    def add_block(self, block_to_add):
        self.return_blockchain_object().append_block(block_to_add)
        print(f"d_{self.idx.split('_')[-1]} - {self.role[0]} has appened a block to its chain. Chain length now - {self.return_blockchain_object().return_chain_length()}")
        # TODO delete has_added_block
        # self.has_added_block = True
        self.the_added_block = block_to_add
        return True

    # also accumulate rewards here
    def process_block(self, block_to_process, log_files_folder_path, conn, conn_cursor, when_resync=False):
        # collect usable updated params, malicious nodes identification, get rewards and do local udpates
        processing_time = time.time()
        # if not self.online_switcher():
        if not self.is_online():
            print(f"{self.role} {self.idx} goes offline when processing the added block. Model not updated and rewards information not upgraded. Outdated information may be obtained by this node if it never resyncs to a different chain.") # may need to set up a flag indicating if a block has been processed
        if block_to_process:
            mined_by = block_to_process.return_mined_by()
            if mined_by in self.black_list:
                # in this system black list is also consistent across devices as it is calculated based on the information on chain, but individual device can decide its own validation/verification mechanisms and has its own
                print(f"The added block is mined by miner {block_to_process.return_mined_by()}, which is in this device's black list. Block will not be processed.")
            else:
                # process validator sig valid transactions
                # used to count positive and negative transactions worker by worker, select the transaction to do global update and identify potential malicious worker
                self_rewards_accumulator = 0
                valid_transactions_records_by_worker = {}
                valid_validator_sig_worker_transacitons_in_block = block_to_process.return_transactions()['valid_validator_sig_transacitons']
                comm_round = block_to_process.return_block_idx()
                self.active_worker_record_by_round[comm_round] = set()
                for valid_validator_sig_worker_transaciton in valid_validator_sig_worker_transacitons_in_block:
                    # verify miner's signature(miner does not get reward for receiving and aggregating)
                    if self.verify_miner_transaction_by_signature(valid_validator_sig_worker_transaciton, mined_by):
                        worker_device_idx = valid_validator_sig_worker_transaciton['worker_device_idx']
                        self.active_worker_record_by_round[comm_round].add(worker_device_idx)
                        if not worker_device_idx in valid_transactions_records_by_worker.keys():
                            valid_transactions_records_by_worker[worker_device_idx] = {}
                            valid_transactions_records_by_worker[worker_device_idx]['positive_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'] = set()
                            valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = None
                        # epoch of this worker's local update
                        local_epoch_seq = valid_validator_sig_worker_transaciton['local_total_accumulated_epochs_this_round']
                        positive_direction_validators = valid_validator_sig_worker_transaciton['positive_direction_validators']
                        negative_direction_validators = valid_validator_sig_worker_transaciton['negative_direction_validators']
                        if len(positive_direction_validators) >= len(negative_direction_validators):
                            # worker transaction can be used
                            valid_transactions_records_by_worker[worker_device_idx]['positive_epochs'].add(local_epoch_seq)
                            valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'].add(local_epoch_seq)
                            # see if this is the latest epoch from this worker
                            if local_epoch_seq == max(valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs']):
                                valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = valid_validator_sig_worker_transaciton['local_updates_params']
                            # give rewards to this worker
                            if self.idx == worker_device_idx:
                                self_rewards_accumulator += valid_validator_sig_worker_transaciton['local_updates_rewards']
                        else:
                            if self.malicious_updates_discount:
                                # worker transaction voted negative and has to be applied for a discount
                                valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'].add(local_epoch_seq)
                                valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'].add(local_epoch_seq)
                                # see if this is the latest epoch from this worker
                                if local_epoch_seq == max(valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs']):
                                    # apply discount
                                    discounted_valid_validator_sig_worker_transaciton_local_updates_params = copy.deepcopy(valid_validator_sig_worker_transaciton['local_updates_params'])
                                    for var in discounted_valid_validator_sig_worker_transaciton_local_updates_params:
                                        discounted_valid_validator_sig_worker_transaciton_local_updates_params[var] *= self.malicious_updates_discount
                                    valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = discounted_valid_validator_sig_worker_transaciton_local_updates_params
                                # worker receive discounted rewards for negative update
                                if self.idx == worker_device_idx:
                                    self_rewards_accumulator += valid_validator_sig_worker_transaciton['local_updates_rewards'] * self.malicious_updates_discount
                            else:
                                # discount specified as 0, worker transaction voted negative and cannot be used
                                valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'].add(local_epoch_seq)
                                # worker does not receive rewards for negative update
                        # give rewards to validators and the miner in this transaction
                        for validator_record in positive_direction_validators + negative_direction_validators:
                            if self.idx == validator_record['validator']:
                                self_rewards_accumulator += validator_record['validation_rewards']
                            if self.idx == validator_record['miner_device_idx']:
                                self_rewards_accumulator += validator_record['miner_rewards_for_this_tx']
                    else:
                        print(f"one validator transaction miner sig found invalid in this block. {self.idx} will drop this block and roll back rewards information")
                        return

                # identify potentially malicious worker
                self.untrustworthy_workers_record_by_comm_round[comm_round] = set()
                for worker_idx, local_updates_direction_records in valid_transactions_records_by_worker.items():
                    if len(local_updates_direction_records['negative_epochs']) >  len(local_updates_direction_records['positive_epochs']):
                        self.untrustworthy_workers_record_by_comm_round[comm_round].add(worker_idx)
                        kick_out_accumulator = 1
                        # check previous rounds
                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_workers_record_by_comm_round.keys():
                                if worker_idx in self.untrustworthy_workers_record_by_comm_round[comm_round_to_check]:
                                    kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:
                            # kick out
                            self.black_list.add(worker_idx)
                            # is it right?
                            if when_resync:
                                msg_end = " when resyncing!\n"
                            else:
                                msg_end = "!\n"
                            if self.devices_dict[worker_idx].return_is_malicious():
                                msg = f"{self.idx} has successfully identified a malicious worker device {worker_idx} in comm_round {comm_round}{msg_end}"
                                with open(f"{log_files_folder_path}/correctly_kicked_workers.txt", 'a') as file:
                                    file.write(msg)
                                conn_cursor.execute("INSERT INTO malicious_workers_log VALUES (?, ?, ?, ?, ?, ?)", (worker_idx, 1, self.idx, "", comm_round, when_resync))
                                conn.commit()
                            else:
                                msg = f"WARNING: {self.idx} has mistakenly regard {worker_idx} as a malicious worker device in comm_round {comm_round}{msg_end}"
                                with open(f"{log_files_folder_path}/mistakenly_kicked_workers.txt", 'a') as file:
                                    file.write(msg)
                                conn_cursor.execute("INSERT INTO malicious_workers_log VALUES (?, ?, ?, ?, ?, ?)", (worker_idx, 0, "", self.idx, comm_round, when_resync))
                                conn.commit()
                            print(msg)

                            # cont = print("Press ENTER to continue")

                # identify potentially compromised validator
                self.untrustworthy_validators_record_by_comm_round[comm_round] = set()
                invalid_validator_sig_worker_transacitons_in_block = block_to_process.return_transactions()['invalid_validator_sig_transacitons']
                for invalid_validator_sig_worker_transaciton in invalid_validator_sig_worker_transacitons_in_block:
                    if self.verify_miner_transaction_by_signature(invalid_validator_sig_worker_transaciton, mined_by):
                        # validator_device_idx = invalid_validator_sig_worker_transaciton['validator']
                        validator_device_idx = invalid_validator_sig_worker_transaciton['validation_done_by']
                        self.untrustworthy_validators_record_by_comm_round[comm_round].add(validator_device_idx)
                        kick_out_accumulator = 1
                        # check previous rounds
                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_validators_record_by_comm_round.keys():
                                if validator_device_idx in self.untrustworthy_validators_record_by_comm_round[comm_round_to_check]:
                                    kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:
                            # kick out
                            self.black_list.add(validator_device_idx)
                            print(f"{validator_device_idx} has been regarded as a compromised validator by {self.idx} in {comm_round}.")
                            # actually, we did not let validator do malicious thing if is_malicious=1 is set to this device. In the submission of 2020/10, we only focus on catching malicious worker
                            # is it right?
                            # if when_resync:
                            #	 msg_end = " when resyncing!\n"
                            # else:
                            #	 msg_end = "!\n"
                            # if self.devices_dict[validator_device_idx].return_is_malicious():
                            #	 msg = f"{self.idx} has successfully identified a compromised validator device {validator_device_idx} in comm_round {comm_round}{msg_end}"
                            #	 with open(f"{log_files_folder_path}/correctly_kicked_validators.txt", 'a') as file:
                            #		 file.write(msg)
                            # else:
                            #	 msg = f"WARNING: {self.idx} has mistakenly regard {validator_device_idx} as a compromised validator device in comm_round {comm_round}{msg_end}"
                            #	 with open(f"{log_files_folder_path}/mistakenly_kicked_validators.txt", 'a') as file:
                            #		 file.write(msg)
                            # print(msg)
                            # cont = print("Press ENTER to continue")
                    else:
                        print(f"one validator transaction miner sig found invalid in this block. {self.idx} will drop this block and roll back rewards information")
                        return
                    # give rewards to the miner in this transaction
                    if self.idx == invalid_validator_sig_worker_transaciton['miner_device_idx']:
                        self_rewards_accumulator += invalid_validator_sig_worker_transaciton['miner_rewards_for_this_tx']
                # miner gets mining rewards
                if self.idx == mined_by:
                    self_rewards_accumulator += block_to_process.return_mining_rewards()
                # set received rewards this round based on info from this block
                self.receive_rewards(self_rewards_accumulator)
                print(f"{self.role} {self.idx} has received total {self_rewards_accumulator} rewards for this comm round.")
                # collect usable worker updates and do global updates
                finally_used_local_params = []
                for worker_device_idx, local_params_record in valid_transactions_records_by_worker.items():
                    if local_params_record['finally_used_params']:
                        # could be None
                        finally_used_local_params.append((worker_device_idx, local_params_record['finally_used_params']))
                # if self.online_switcher():
                if self.is_online():
                    self.global_update(finally_used_local_params)
                else:
                    print(f"Unfortunately, {self.role} {self.idx} goes offline when it's doing global_updates.")
        processing_time = (time.time() - processing_time)/self.computation_power
        return processing_time

    def add_to_round_end_time(self, time_to_add):
        self.round_end_time += time_to_add


    def other_tasks_at_the_end_of_comm_round(self, this_comm_round, log_files_folder_path):
        self.kick_out_slow_or_lazy_workers(this_comm_round, log_files_folder_path)

    def kick_out_slow_or_lazy_workers(self, this_comm_round, log_files_folder_path):
        for device in self.peer_list:
            if device.return_role() == 'worker':
                if this_comm_round in self.active_worker_record_by_round.keys():
                    if not device.return_idx() in self.active_worker_record_by_round[this_comm_round]:
                        not_active_accumulator = 1
                        # check if not active for the past (lazy_worker_knock_out_rounds - 1) rounds
                        for comm_round_to_check in range(this_comm_round - self.lazy_worker_knock_out_rounds + 1, this_comm_round):
                            if comm_round_to_check in self.active_worker_record_by_round.keys():
                                if not device.return_idx() in self.active_worker_record_by_round[comm_round_to_check]:
                                    not_active_accumulator += 1
                        if not_active_accumulator == self.lazy_worker_knock_out_rounds:
                            # kick out
                            self.black_list.add(device.return_idx())
                            msg = f"worker {device.return_idx()} has been regarded as a lazy worker by {self.idx} in comm_round {this_comm_round}.\n"
                            with open(f"{log_files_folder_path}/kicked_lazy_workers.txt", 'a') as file:
                                file.write(msg)
                else:
                    # this may happen when a device is put into black list by every worker in a certain comm round
                    pass

    def update_model_after_chain_resync(self, log_files_folder_path, conn, conn_cursor):
        # reset global params to the initial weights of the net
        self.global_parameters = copy.deepcopy(self.initial_net_parameters)
        # in future version, develop efficient updating algorithm based on chain difference
        for block in self.return_blockchain_object().return_chain_structure():
            self.process_block(block, log_files_folder_path, conn, conn_cursor, when_resync=True)

    def return_pow_difficulty(self):
        return self.pow_difficulty

    def register_in_the_network(self, check_online=False):
        if self.aio:
            self.add_peers(set(self.devices_dict.values()))
        else:
            potential_registrars = set(self.devices_dict.values())
            # it cannot register with itself
            potential_registrars.discard(self)
            # pick a registrar
            registrar = random.sample(potential_registrars, 1)[0]
            if check_online:
                if not registrar.is_online():
                    online_registrars = set()
                    for registrar in potential_registrars:
                        if registrar.is_online():
                            online_registrars.add(registrar)
                    if not online_registrars:
                        return False
                    registrar = random.sample(online_registrars, 1)[0]
            # registrant add registrar to its peer list
            self.add_peers(registrar)
            # this device sucks in registrar's peer list
            self.add_peers(registrar.return_peers())
            # registrar adds registrant(must in this order, or registrant will add itself from registrar's peer list)
            registrar.add_peers(self)
            return True

    ''' Worker '''
    def malicious_worker_add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                noise = self.noise_variance * torch.randn(m.weight.size())
                variance_of_noise = torch.var(noise)
                m.weight.add_(noise.to(self.dev))
                self.variance_of_noises.append(float(variance_of_noise))

    # TODO change to computation power
    def worker_local_update(self, rewards, log_files_folder_path_comm_round, comm_round, local_epochs=1):
        print(f"Worker {self.idx} is doing local_update with computation power {self.computation_power} and link speed {round(self.link_speed,3)} bytes/s")
        self.net.load_state_dict(self.global_parameters, strict=True)
        self.local_update_time = time.time()
        is_malicious_node = "M" if self.return_is_malicious() else "B"
        self.local_updates_rewards_per_transaction = 0
        losses = []
        for epoch in range(local_epochs):
            running_loss = 0.0
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)   # Compute prediction error  # print("Preds", preds)
                loss = self.loss_func(preds, label)
                loss.backward()  #Backpropagation    # print("Loss", loss.item())
                self.opti.step()
                self.opti.zero_grad()
                running_loss += loss.item() * label.size(0)
                print("Size of Data DEV:",label.shape[0])
                losses.append(running_loss/len(self.train_dl))
                self.local_updates_rewards_per_transaction += (rewards * label.shape[0])
            self.local_total_epoch += 1
        try:
            self.local_update_time = (time.time() - self.local_update_time)/self.computation_power
        except:
            self.local_update_time = float('inf')
        if self.is_malicious:
            self.net.apply(self.malicious_worker_add_noise_to_weights)
            print(f"-----------------------------------------------malicious worker {self.idx} has added noise to its local updated weights before transmitting")
            with open(f"{log_files_folder_path_comm_round}/comm_{comm_round}_variance_of_noises.txt", "a") as file:
                file.write(f"{self.return_idx()} {self.return_role()} {is_malicious_node} noise variances: {self.variance_of_noises}\n")
        print(f"Done {local_epochs} epoch(s) and total {self.local_total_epoch} epochs")
        self.local_train_parameters = self.net.state_dict()
        return self.local_update_time

    # used to simulate time waste when worker goes offline during transmission to validator
    def waste_one_epoch_local_update_time(self, opti):
        if self.computation_power == 0:
            return float('inf'), None
        else:
            validation_net = copy.deepcopy(self.net)
            currently_used_lr = 0.01
            for param_group in self.opti.param_groups:
                currently_used_lr = param_group['lr']
            # by default use SGD. Did not implement others
            if opti == 'SGD':
                validation_opti = optim.SGD(validation_net.parameters(), lr=currently_used_lr)
            local_update_time = time.time()
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = validation_net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                validation_opti.step()
                validation_opti.zero_grad()
            return (time.time() - local_update_time)/self.computation_power, validation_net.state_dict()

    def set_accuracy_this_round(self, accuracy):
        self.accuracy_this_round = accuracy

    def return_accuracy_this_round(self):
        return self.accuracy_this_round

    def return_link_speed(self):
        return self.link_speed

    def return_local_updates_and_signature(self, comm_round):
        self.check_xmss_tree_index()
        # local_total_accumulated_epochs_this_round also stands for the lastest_epoch_seq for this transaction(local params are calculated after this amount of local epochs in this round)
        # last_local_iteration(s)_spent_time may be recorded to determine calculating time? But what if nodes do not wish to disclose its computation power
        local_updates_dict = {'worker_device_idx': self.idx, 'in_round_number': comm_round,
                              "local_updates_params": copy.deepcopy(self.local_train_parameters),
                              "local_updates_rewards": self.local_updates_rewards_per_transaction,
                              "local_iteration(s)_spent_time": self.local_update_time,
                              "local_total_accumulated_epochs_this_round": self.local_total_epoch,
                              #"worker_rsa_pub_key": self.return_rsa_pub_key()
                              "worker_xmss_pub_key": self.return_xmss_pub_key(),
                              "worker_tree_no": self.treeNo
                              }

        # print("Worker Device ID",self.idx)
        # print("Worker Tree No", self.treeNo)
        # print("Public Key", self.return_xmss_pub_key())

        # print(local_updates_dict.keys())
        # print(type(local_updates_dict))
        # print("999999999. ===============WORKER 1 Local UPDATES=======================",
        #       int.from_bytes(sha256(str(sorted(local_updates_dict.items())).encode("utf-8")).digest(),
        #                      byteorder='big'))
        local_updates_dict["worker_signature"] = self.sign_msg_xmss(sorted(local_updates_dict.items()))
        # print("999999999. ===============WORKER 2 Local UPDATES=======================",
        #      int.from_bytes(sha256(str(sorted(local_updates_dict.items())).encode("utf-8")).digest(),
        #                     byteorder='big'))
        return local_updates_dict

    def worker_reset_vars_for_new_round(self):
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.local_updates_rewards_per_transaction = 0
        self.has_added_block = False
        self.the_added_block = None
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_total_epoch = 0
        self.variance_of_noises.clear()
        self.round_end_time = 0

    def receive_block_from_miner(self, received_block, source_miner):
        if not (received_block.return_mined_by() in self.black_list or source_miner in self.black_list):
            self.received_block_from_miner = copy.deepcopy(received_block)
            print(f"{self.role} {self.idx} has received a new block from {source_miner} mined by {received_block.return_mined_by()}.")
        else:
            print(f"Either the block sending miner {source_miner} or the miner {received_block.return_mined_by()} mined this block is in worker {self.idx}'s black list. Block is not accepted.")


    def toss_received_block(self):
        self.received_block_from_miner = None

    def return_received_block_from_miner(self):
        return self.received_block_from_miner

    def validate_model_weights(self, weights_to_eval=None):
        with torch.no_grad():
            if weights_to_eval:
                self.net.load_state_dict(weights_to_eval, strict=True)
            else:
                self.net.load_state_dict(self.global_parameters, strict=True)
            sum_accu = 0
            num = 0
            valid_loss = 0.0
            losses = []
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = self.loss_func(preds, label)  #DEV ADDED

                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1

            # print(sum_accu / num)

            # return float(sum_accu / num), losses
            return sum_accu / num, losses



    def validate_model_weights1(self, weights_to_eval, deviceId, comm_round):
        with torch.no_grad():
            if weights_to_eval:
                self.net.load_state_dict(weights_to_eval, strict=True)

            sum_accu = 0
            num = 0
            valid_loss = 0.0
            losses = []
            for data, label in self.test_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = self.loss_func(preds, label)  #DEV ADDED

                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
                # if weights_to_eval:
                print("validation Loss:", loss.item())
                valid_loss += loss.item() * data.size(0)
                losses.append(valid_loss / len(self.test_dl))
                print("NUM============", num)
            # epoch_loss = valid_loss/

            with open(f"{bench_folder}/validation.txt","a") as file:
                file.write(f"COMM ROUND: {comm_round} | Validation done by {self.return_idx()} for worker {deviceId}  which is  {str(self.devices_dict[deviceId].is_malicious)}\n")
                file.write(f"{losses}\n")
            # plt.scatter(range(len(losses)), losses)
            # print("Validation Lossess:",losses)
            # plt.title()
            # # plt.show()
            # print(sum_accu / num)
            return float(sum_accu / num), losses


    def global_update(self, local_update_params_potentially_to_be_used):
        # filter local_params
        local_params_by_benign_workers = []
        for (worker_device_idx, local_params) in local_update_params_potentially_to_be_used:
            if not worker_device_idx in self.black_list:
                local_params_by_benign_workers.append(local_params)
            else:
                print(f"global update skipped for a worker {worker_device_idx} in {self.idx}'s black list")
        if local_params_by_benign_workers:
            # avg the gradients
            sum_parameters = None
            for local_updates_params in local_params_by_benign_workers:
                if sum_parameters is None:
                    sum_parameters = copy.deepcopy(local_updates_params)
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += local_updates_params[var]
            # number of finally filtered workers' updates
            num_participants = len(local_params_by_benign_workers)
            for var in self.global_parameters:
                self.global_parameters[var] = (sum_parameters[var] / num_participants)
            print(f"global updates done by {self.idx}")
        else:
            print(f"There are no available local params for {self.idx} to perform global updates in this comm round.")



    ''' miner '''

    def request_to_download(self, block_to_download, requesting_time_point):
        print(f"miner {self.idx} is requesting its associated devices to download the block it just added to its chain")
        devices_in_association = self.miner_associated_validator_set.union(self.miner_associated_worker_set)
        for device in devices_in_association:
            # theoratically, one device is associated to a specific miner, so we don't have a miner_block_arrival_queue here
            # if self.online_switcher() and device.online_switcher():
            if self.is_online() and device.is_online():
                miner_link_speed = self.return_link_speed()
                device_link_speed = device.return_link_speed()
                lower_link_speed = device_link_speed if device_link_speed < miner_link_speed else miner_link_speed
                transmission_delay = getsizeof(str(block_to_download.__dict__))/lower_link_speed
                verified_block, verification_time = device.verify_block(block_to_download, block_to_download.return_mined_by())
                if verified_block:
                    # forgot to check for maliciousness of the block miner
                    device.add_block(verified_block)
                device.add_to_round_end_time(requesting_time_point + transmission_delay + verification_time)
            else:
                print(f"Unfortunately, either miner {self.idx} or {device.return_idx()} goes offline while processing this request-to-download block.")
    # def request_to_download(self, block_to_download, ):
    #     print(f"miner {self.idx} is requesting its associated devices to download the block it just added to its chain")
    #     devices_in_association = self.miner_associated_validator_set.union(self.miner_associated_worker_set)
    #     for device in devices_in_association:
    #         # theoratically, one device is associated to a specific miner, so we don't have a miner_block_arrival_queue here
    #         # if self.online_switcher() and device.online_switcher():
    #         if self.is_online() and device.is_online():
    #             miner_link_speed = self.return_link_speed()
    #             device_link_speed = device.return_link_speed()
    #             lower_link_speed = device_link_speed if device_link_speed < miner_link_speed else miner_link_speed
    #             transmission_delay = getsizeof(str(block_to_download.__dict__))/lower_link_speed
    #             verified_block, verification_time = device.verify_block(block_to_download, block_to_download.return_mined_by())
    #             if verified_block:
    #                 # forgot to check for maliciousness of the block miner
    #                 device.add_block(verified_block)
    #             device.add_to_round_end_time(transmission_delay + verification_time)
    #             # device.add_to_round_end_time(requesting_time_point + transmission_delay + verification_time)
    #         else:
    #             print(f"Unfortunately, either miner {self.idx} or {device.return_idx()} goes offline while processing this request-to-download block.")

    def propagated_the_block(self, propagating_time_point, block_to_propagate):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(f"{self.role} {self.idx} is propagating its mined block to {peer.return_role()} {peer.return_idx()}.")
                        # if peer.online_switcher():
                        if peer.is_online():
                            peer.accept_the_propagated_block(self, self.block_generation_time_point, block_to_propagate)
                    else:
                        print(f"Destination miner {peer.return_idx()} is in {self.role} {self.idx}'s black_list. Propagating skipped for this dest miner.")
   
    def accept_the_propagated_block(self, source_miner, source_miner_propagating_time_point, propagated_block):
        if not source_miner.return_idx() in self.black_list:
            source_miner_link_speed = source_miner.return_link_speed()
            this_miner_link_speed = self.link_speed
            lower_link_speed = this_miner_link_speed if this_miner_link_speed < source_miner_link_speed else source_miner_link_speed
            transmission_delay = getsizeof(str(propagated_block.__dict__))/lower_link_speed
            self.unordered_propagated_block_processing_queue[source_miner_propagating_time_point + transmission_delay] = propagated_block
            print(f"{self.role} {self.idx} has accepted accepted a propagated block from miner {source_miner.return_idx()}")
        else:
            print(f"Source miner {source_miner.return_role()} {source_miner.return_idx()} is in {self.role} {self.idx}'s black list. Propagated block not accepted.")

    def add_propagated_block_to_processing_queue(self, arrival_time, propagated_block):
        self.unordered_propagated_block_processing_queue[arrival_time] = propagated_block

    def return_unordered_propagated_block_processing_queue(self):
        return self.unordered_propagated_block_processing_queue

    def return_associated_validators(self):
        return self.miner_associated_validator_set

    def return_miner_acception_wait_time(self):
        return self.miner_acception_wait_time

    def return_miner_accepted_transactions_size_limit(self):
        return self.miner_accepted_transactions_size_limit

    def return_miners_eligible_to_continue(self):
        miners_set = set()
        for peer in self.peer_list:
            if peer.return_role() == 'miner':
                miners_set.add(peer)
        miners_set.add(self)
        return miners_set

    def return_accepted_broadcasted_transactions(self):
        return self.broadcasted_transactions

    def verify_validator_transaction(self, transaction_to_verify, rewards):

        if self.computation_power == 0:
            print(f"miner {self.idx} has computation power 0 and will not be able to verify this transaction in time")
            return False, None
        else:
            transaction_validator_idx = transaction_to_verify['validation_done_by']
            if transaction_validator_idx in self.black_list:
                print(f"{transaction_validator_idx} is in miner's blacklist. Trasaction won't get verified.")
                return False, None
            verification_time = time.time()

            # print("14. ===============VALIDATOR  1  =======================",
            #       int.from_bytes(sha256(str(sorted(transaction_to_verify.items())).encode("utf-8")).digest(), byteorder='big'))

            # print("1. ========VALIDATOR 1==============================",
            #       int.from_bytes(sha256(json.dump(transaction_to_verify, sort_keys=True)), byteorder='big'))



            # print(transaction_to_verify.keys())
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(transaction_to_verify)
                del transaction_before_signed["validator_signature"]
                # transaction_before_signed = copy.deepcopy(transaction_to_verify)
                # print("1. ========VALIDATOR 2 ==============================",
                #       int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode("utf-8")).digest(),
                #                      byteorder='big'))


                # del transaction_before_signed["xmss_tree_validator"]
                # modulus = transaction_to_verify['validator_rsa_pub_key']["modulus"]
                # pub_key = transaction_to_verify['validator_rsa_pub_key']["pub_key"]
                pub_key = transaction_to_verify['validator_xmss_pub_key']["pub_key"]
                signature = transaction_to_verify['validator_signature']
                tree_no = transaction_to_verify['validator_tree_no']
                id = transaction_to_verify['validation_done_by']
                # print(type(tree_no))
                # print(tree_no)
                # data = json.load(transaction_before_signed)

                # df = pd.DataFrame(data)
                # print(df)

                # print(len(transaction_before_signed))
                # print(type(transaction_before_signed))
                # print(transaction_before_signed.keys())

                # print("7 --------------------------", transaction_before_signed)

                # print(transaction_before_signed.keys())
                # print(len(transaction_before_signed))
                # print("14. ===============VALIDATOR 2 HEXDIGEST=======================",
                #       sha256(str(sorted(transaction_before_signed.items())).encode("utf-8")).digest())

                # print("1. ========VALIDATOR 3 ==============================",
                #       int.from_bytes(sha256(str(sorted(transaction_to_verify.items())).encode("utf-8")).digest(),
                #                      byteorder='big'))
                #
                # print("1. ========VALIDATOR 3 ==============================",
                #       int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode("utf-8")).digest(),
                #                      byteorder='big'))

                # hash = sorted(transaction_before_signed.items())
                hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
                # print("VALIDATOR SIGNING TRANSACTIONS>>>>>>>>>>>>>>>>>>>>>")
                # print("22. !!!!!!!!!!!!!!!!!!!!!!BEFORE SIGNED IN VERIFY -------------",
                #       sha256(str(sorted(transaction_before_signed.items())).encode("utf-8")).digest())
                # print("22222222222222222.........",transaction_before_signed.items())

                # begin verification
                # hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
                # hashFromSignature = pow(signature, pub_key, modulus)
                # if hash == hashFromSignature:
                # print("VALIDATOR TRANSACTIONS being VERIFIED>>>>>>>>>>>>>>>>>>>>>")
                # print(xmss_tree.public_key)
                if self.verify_msg_xmss(hash, signature, pub_key) and self.verify_dilithium(pub_key, id, tree_no):
                    self.receive_rewards(rewards)
                    print(
                        f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.idx}!")
                    verification_time = (time.time() - verification_time) / self.computation_power
                    return verification_time, True
                else:
                    print(f"Signature invalid. Transaction from validator {transaction_validator_idx} is NOT verified.")
                    return (time.time() - verification_time) / self.computation_power, False
            else:
                print(
                    f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.idx}!")
                verification_time = (time.time() - verification_time) / self.computation_power
                return verification_time, True

    def sign_candidate_transaction(self, candidate_transaction):
        self.check_xmss_tree_index()
        signing_time = time.time()
        # candidate_transaction['miner_rsa_pub_key'] = self.return_rsa_pub_key()
        candidate_transaction['miner_xmss_pub_key'] = self.return_xmss_pub_key()
        candidate_transaction['miner_tree_no'] = self.return_tree_no()
        if 'miner_signature' in candidate_transaction.keys():
            del candidate_transaction['miner_signature']
        candidate_transaction["miner_signature"] = self.sign_msg_xmss(sorted(candidate_transaction.items()))
        signing_time = (time.time() - signing_time) / self.computation_power
        return signing_time

    def mine_block(self, candidate_block, rewards, starting_nonce=0):
        candidate_block.set_mined_by(self.idx)
        pow_mined_block = self.proof_of_work(candidate_block)
        # pow_mined_block.set_mined_by(self.idx)
        pow_mined_block.set_mining_rewards(3 * rewards)
        return pow_mined_block

    def proof_of_work(self, candidate_block, starting_nonce=0):
        candidate_block.set_mined_by(self.idx)
        ''' Brute Force the nonce '''
        candidate_block.set_nonce(starting_nonce)
        current_hash = candidate_block.compute_hash()
        # candidate_block.set_pow_difficulty(self.pow_difficulty)
        while not current_hash.startswith('0' * self.pow_difficulty):
            candidate_block.nonce_increment()
            current_hash = candidate_block.compute_hash()
        # return the qualified hash as a PoW proof, to be verified by other devices before adding the block
        # also set its hash as well. block_hash is the same as pow proof
        candidate_block.set_pow_proof(current_hash)
        return candidate_block

    def set_block_generation_time_point(self, block_generation_time_point):
        self.block_generation_time_point = block_generation_time_point

    def return_block_generation_time_point(self):
        return self.block_generation_time_point

    def receive_propagated_block(self, received_propagated_block):
        if not received_propagated_block.return_mined_by() in self.black_list:
            self.received_propagated_block = copy.deepcopy(received_propagated_block)
            print(f"Miner {self.idx} has received a propagated block from {received_propagated_block.return_mined_by()}.")
        else:
            print(f"Propagated block miner {received_propagated_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")

    def receive_propagated_validator_block(self, received_propagated_validator_block):
        if not received_propagated_validator_block.return_mined_by() in self.black_list:
            self.received_propagated_validator_block = copy.deepcopy(received_propagated_validator_block)
            print(f"Miner {self.idx} has received a propagated validator block from {received_propagated_validator_block.return_mined_by()}.")
        else:
            print(f"Propagated validator block miner {received_propagated_validator_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")

    def return_propagated_block(self):
        return self.received_propagated_block

    def return_propagated_validator_block(self):
        return self.received_propagated_validator_block

    def toss_propagated_block(self):
        self.received_propagated_block = None

    def toss_ropagated_validator_block(self):
        self.received_propagated_validator_block = None

    def miner_reset_vars_for_new_round(self):
        self.miner_associated_worker_set.clear()
        self.miner_associated_validator_set.clear()
        self.unconfirmmed_transactions.clear()
        self.broadcasted_transactions.clear()
        # self.unconfirmmed_validator_transactions.clear()
        # self.validator_accepted_broadcasted_worker_transactions.clear()
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.has_added_block = False
        self.the_added_block = None
        self.unordered_arrival_time_accepted_validator_transactions.clear()
        self.miner_accepted_broadcasted_validator_transactions.clear()
        self.block_generation_time_point = None
#		self.block_to_add = None
        self.unordered_propagated_block_processing_queue.clear()
        self.round_end_time = 0

    def set_unordered_arrival_time_accepted_validator_transactions(self, unordered_arrival_time_accepted_validator_transactions):
        self.unordered_arrival_time_accepted_validator_transactions = unordered_arrival_time_accepted_validator_transactions

    def return_unordered_arrival_time_accepted_validator_transactions(self):
        return self.unordered_arrival_time_accepted_validator_transactions

    def miner_broadcast_validator_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(f"miner {self.idx} is broadcasting received validator transactions to miner {peer.return_idx()}.")
                        final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner = copy.copy(self.unordered_arrival_time_accepted_validator_transactions)
                        # offline situation similar in validator_broadcast_worker_transactions()
                        for arrival_time, tx in self.unordered_arrival_time_accepted_validator_transactions.items():
                            # if not (self.online_switcher() and peer.online_switcher()):
                            if not (self.is_online() and peer.is_online()):
                                del final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner[arrival_time]
                        peer.accept_miner_broadcasted_validator_transactions(self, final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)
                        print(f"miner {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)} validator transactions to miner {peer.return_idx()}.")
                    else:
                        print(f"Destination miner {peer.return_idx()} is in miner {self.idx}'s black_list. broadcasting skipped for this dest miner.")

    def accept_miner_broadcasted_validator_transactions(self, source_device, unordered_transaction_arrival_queue_from_source_miner):
        # discard malicious node
        if not source_device.return_idx() in self.black_list:
            self.miner_accepted_broadcasted_validator_transactions.append({'source_device_link_speed': source_device.return_link_speed(),'broadcasted_transactions': copy.deepcopy(unordered_transaction_arrival_queue_from_source_miner)})
            print(f"{self.role} {self.idx} has accepted validator transactions from {source_device.return_role()} {source_device.return_idx()}")
        else:
            print(f"Source miner {source_device.return_role()} {source_device.return_idx()} is in {self.role} {self.idx}'s black list. Broadcasted transactions not accepted.")

    def return_accepted_broadcasted_validator_transactions(self):
        return self.miner_accepted_broadcasted_validator_transactions

    def set_candidate_transactions_for_final_mining_queue(self, final_transactions_arrival_queue):
        self.final_candidate_transactions_queue_to_mine = final_transactions_arrival_queue

    def return_final_candidate_transactions_mining_queue(self):
        return self.final_candidate_transactions_queue_to_mine

    ''' validator '''
    def validator_reset_vars_for_new_round(self):
        self.validation_rewards_this_round = 0
        # self.accuracies_this_round = {}
        self.has_added_block = False
        self.the_added_block = None
        self.validator_associated_miner = None
        self.validator_local_accuracy = None
        self.validator_associated_worker_set.clear()
        #self.post_validation_transactions.clear()
        #self.broadcasted_post_validation_transactions.clear()
        self.unordered_arrival_time_accepted_worker_transactions.clear()
        self.final_transactions_queue_to_validate.clear()
        self.validator_accepted_broadcasted_worker_transactions.clear()
        self.post_validation_transactions_queue.clear()
        self.round_end_time = 0

    def add_post_validation_transaction_to_queue(self, transaction_to_add):
        self.post_validation_transactions_queue.append(transaction_to_add)

    def return_post_validation_transactions_queue(self):
        return self.post_validation_transactions_queue

    def return_online_workers(self):
        online_workers_in_peer_list = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "worker":
                    online_workers_in_peer_list.add(peer)
        return online_workers_in_peer_list

    def return_validations_and_signature(self, comm_round):
        self.check_xmss_tree_index()
        # print("This is function --------____ +++++++++++++++++++++++++++++++++=N EN VD LK VDLKJV ")
        validation_transaction_dict = {'validator_device_idx': self.idx, 'round_number': comm_round,
                                       'accuracies_this_round': copy.deepcopy(self.accuracies_this_round),
                                       'validation_effort_rewards': self.validation_rewards_this_round,
                                       # "rsa_pub_key": self.return_rsa_pub_key()
                                       "validator_xmss_pub_key": self.return_xmss_pub_key(),
                                       "validator_tree_no": self.treeNo,
                                       }
        validation_transaction_dict["validator_signature"] = self.sign_msg_xmss(sorted(validation_transaction_dict.items()))
        # validation_transaction_dict["xmss_tree_validator"] = self.xmss_tree
        return validation_transaction_dict

    def add_worker_to_association(self, worker_device):
        if not worker_device.return_idx() in self.black_list:
            self.associated_worker_set.add(worker_device)
        else:
            print(f"WARNING: {worker_device.return_idx()} in validator {self.idx}'s black list. Not added by the validator.")

    def associate_with_miner(self):
        miners_in_peer_list = set()
        for peer in self.peer_list:
            if peer.return_role() == "miner":
                if not peer.return_idx() in self.black_list:
                    miners_in_peer_list.add(peer)
        if not miners_in_peer_list:
            return False
        self.validator_associated_miner = random.sample(miners_in_peer_list, 1)[0]
        return self.validator_associated_miner


    ''' miner and validator '''
    def add_device_to_association(self, to_add_device):
        if not to_add_device.return_idx() in self.black_list:
            vars(self)[f'{self.role}_associated_{to_add_device.return_role()}_set'].add(to_add_device)
        else:
            print(f"WARNING: {to_add_device.return_idx()} in {self.role} {self.idx}'s black list. Not added by the {self.role}.")

    def return_associated_workers(self):
        return vars(self)[f'{self.role}_associated_worker_set']

    def sign_block(self, block_to_sign):
        self.check_xmss_tree_index()
        block_to_sign.set_tree_no(self.treeNo)
        block_to_sign.set_signature(self.sign_msg_xmss(block_to_sign.__dict__))

    def add_unconfirmmed_transaction(self, unconfirmmed_transaction, souce_device_idx):
        if not souce_device_idx in self.black_list:
            self.unconfirmmed_transactions.append(copy.deepcopy(unconfirmmed_transaction))
            print(f"{souce_device_idx}'s transaction has been recorded by {self.role} {self.idx}")
        else:
            print(f"Source device {souce_device_idx} is in the black list of {self.role} {self.idx}. Transaction has not been recorded.")

    def return_unconfirmmed_transactions(self):
        return self.unconfirmmed_transactions

    def broadcast_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == self.role:
                    if not peer.return_idx() in self.black_list:
                        print(f"{self.role} {self.idx} is broadcasting transactions to {peer.return_role()} {peer.return_idx()}.")
                        peer.accept_broadcasted_transactions(self, self.unconfirmmed_transactions)
                    else:
                        print(f"Destination {peer.return_role()} {peer.return_idx()} is in {self.role} {self.idx}'s black_list. broadcasting skipped.")

    def accept_broadcasted_transactions(self, source_device, broadcasted_transactions):
        # discard malicious node
        if not source_device.return_idx() in self.black_list:
            self.broadcasted_transactions.append(copy.deepcopy(broadcasted_transactions))
            print(f"{self.role} {self.idx} has accepted transactions from {source_device.return_role()} {source_device.return_idx()}")
        else:
            print(f"Source {source_device.return_role()} {source_device.return_idx()} is in {self.role} {self.idx}'s black list. Transaction not accepted.")

    ''' worker and validator '''

    def set_mined_block(self, mined_block):
        self.mined_block = mined_block

    def return_mined_block(self):
        return self.mined_block

    def associate_with_device(self, to_associate_device_role):
        to_associate_device = vars(self)[f'{self.role}_associated_{to_associate_device_role}']
        shuffled_peer_list = list(self.peer_list)
        random.shuffle(shuffled_peer_list)
        for peer in shuffled_peer_list:
            # select the first found eligible device from a shuffled order
            if peer.return_role() == to_associate_device_role and peer.is_online():
                if not peer.return_idx() in self.black_list:
                    to_associate_device = peer
        if not to_associate_device:
            # there is no device matching the required associated role in this device's peer list
            return False
        print(f"{self.role} {self.idx} associated with {to_associate_device.return_role()} {to_associate_device.return_idx()}")
        return to_associate_device

    ''' validator '''

    def set_unordered_arrival_time_accepted_worker_transactions(self, unordered_transaction_arrival_queue):
        self.unordered_arrival_time_accepted_worker_transactions = unordered_transaction_arrival_queue

    def return_unordered_arrival_time_accepted_worker_transactions(self):
        return self.unordered_arrival_time_accepted_worker_transactions

    def validator_broadcast_worker_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "validator":
                    if not peer.return_idx() in self.black_list:
                        print(f"validator {self.idx} is broadcasting received validator transactions to validator {peer.return_idx()}.")
                        final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator = copy.copy(self.unordered_arrival_time_accepted_worker_transactions)
                        # if offline, it's like the broadcasted transaction was not received, so skip a transaction
                        for arrival_time, tx in self.unordered_arrival_time_accepted_worker_transactions.items():
                            # if not (self.online_switcher() and peer.online_switcher()):
                            if not (self.is_online() and peer.is_online()):
                                del final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator[arrival_time]
                        # in the real distributed system, it should be broadcasting transaction one by one. Here we send the all received transactions(while online) and later calculate the order for the individual broadcasting transaction's arrival time mixed with the transactions itself received
                        peer.accept_validator_broadcasted_worker_transactions(self, final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator)
                        print(f"validator {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator)} worker transactions to validator {peer.return_idx()}.")
                    else:
                        print(f"Destination validator {peer.return_idx()} is in this validator {self.idx}'s black_list. broadcasting skipped for this dest validator.")

    def accept_validator_broadcasted_worker_transactions(self, source_validator,
                                                         unordered_transaction_arrival_queue_from_source_validator):
        if not source_validator.return_idx() in self.black_list:
            self.validator_accepted_broadcasted_worker_transactions.append(
                {'source_validator_link_speed': source_validator.return_link_speed(),
                 'broadcasted_transactions': copy.deepcopy(unordered_transaction_arrival_queue_from_source_validator)})
            print(
                f"validator {self.idx} has accepted worker transactions from validator {source_validator.return_idx()}")
        else:
            print(
                f"Source validator {source_validator.return_idx()} is in validator {self.idx}'s black list. Broadcasted transactions not accepted.")

    def return_accepted_broadcasted_worker_transactions(self):
        return self.validator_accepted_broadcasted_worker_transactions

    def set_transaction_for_final_validating_queue(self, final_transactions_arrival_queue):
        self.final_transactions_queue_to_validate = final_transactions_arrival_queue

    def return_final_transactions_validating_queue(self):
        return self.final_transactions_queue_to_validate

    def validator_update_model_by_one_epoch_and_validate_local_accuracy(self, opti):
        # return time spent
        print(f"validator {self.idx} is performing one epoch of local update and validation")
        if self.computation_power == 0:
            print(f"validator {self.idx} has computation power 0 and will not be able to complete this validation")
            return float('inf')
        else:
            updated_net = copy.deepcopy(self.net)
            currently_used_lr = 0.01
            for param_group in self.opti.param_groups:
                currently_used_lr = param_group['lr']
            # by default use SGD. Did not implement others
            if opti == 'SGD':
                validation_opti = optim.SGD(updated_net.parameters(), lr=currently_used_lr)
            local_validation_time = time.time()
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = updated_net(data)
                loss = self.loss_func(preds, label)
                loss.backward()
                validation_opti.step()
                validation_opti.zero_grad()
            # validate by local test set
            with torch.no_grad():
                sum_accu = 0
                num = 0
                for data, label in self.test_dl:
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = updated_net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
            self.validator_local_accuracy = sum_accu / num
            print(f"validator {self.idx} locally updated model has accuracy {self.validator_local_accuracy} on its local test set")
            return (time.time() - local_validation_time)/self.computation_power

    # TODO validator_threshold
    def validate_worker_transaction(self, transaction_to_validate, rewards, log_files_folder_path, comm_round,
                                    malicious_validator_on, average_accuracies):
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if self.computation_power == 0:
            print(
                f"validator {self.idx} has computation power 0 and will not be able to validate this transaction in time")
            return False, False
        else:
            worker_transaction_device_idx = transaction_to_validate['worker_device_idx']
            if worker_transaction_device_idx in self.black_list:
                print(f"{worker_transaction_device_idx} is in validator's blacklist. Trasaction won't get validated.")
                return False, False
            validation_time = time.time()
            #
            # print("999999999. ===============WORKER  2=======================",
            #       int.from_bytes(sha256(str(sorted(transaction_to_validate.items())).encode("utf-8")).digest(),
            #                      byteorder='big'))

            if self.check_signature:
                transaction_before_signed = copy.deepcopy(transaction_to_validate)
                del transaction_before_signed["worker_signature"]
                # modulus = transaction_to_validate['worker_rsa_pub_key']["modulus"]
                pub_key = transaction_to_validate['worker_xmss_pub_key']["pub_key"]
                signature = transaction_to_validate["worker_signature"]
                worker_tree_no = transaction_to_validate["worker_tree_no"]
                worker_device_id = transaction_to_validate["worker_device_idx"]
                # print("Worker Device ID", worker_device_id)
                # print("Being Verified by Device", self.idx)
                # print("Worker Tree No", worker_tree_no)
                # print("Public Key", pub_key)

                #For Dilithium Verification
                # hash_pk_xmss = int.from_bytes(sha256(str(pub_key).encode("utf-8")).digest(), byteorder='big')
                # dilithium_signature = self.dilithium_signatures[worker_tree_no]
                # print("DILITHIUM DETAILS>>>>>>>>>>>>>>", self.dilithium_signatures[worker_tree_no])
                # print(type(dilithium_signature))


                # try:
                #     self.verify_dilithium(pub_key, dilithium_signature, self.return_dilithium_pub_key())
                #     # verify_dilithium = oqs.Signature("Dilithium5").verify(str(hash_pk_xmss).encode("utf-8"), dilithium_signature, self.dilithium_public_key)
                #     print("Dilithium Signature Verification")
                # except:
                #     print("Verify Worker Dilithium -----------------", "ERROR")

                # begin validation
                # 1 - verify signature
                # print(transaction_before_signed)

                # print("999999999. ===============WORKER  1=======================",
                #       int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode("utf-8")).digest(),
                #                      byteorder='big'))

                # hash = sorted(transaction_before_signed.items())
                hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
                # hash1 = int.from_bytes(sha256(str(pub_key).encode('utf-8')).digest(), byteorder='big')


                # print("Signature Hash..Verify", sha256(str(signature).encode("utf-8")).digest())

                # hash = int.from_bytes(sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
                # hashFromSignature = pow(signature, pub_key, modulus)
                # if hash == hashFromSignature:
                # print("Public key ,,,,,,,,,Verifying", xmss_tree.public_key )
                if self.verify_msg_xmss(hash, signature, pub_key) and self.verify_dilithium(pub_key, worker_device_id, worker_tree_no):
                    print(
                        f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {self.idx}!")
                    transaction_to_validate['worker_signature_valid'] = True
                else:
                    print(
                        f"Signature invalid. Transaction from worker {worker_transaction_device_idx} does NOT pass verification.")
                    # will also add sig not verified transaction due to the validator's verification effort and its rewards needs to be recorded in the block
                    transaction_to_validate['worker_signature_valid'] = False
            else:
                print(
                    f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {self.idx}!")
                transaction_to_validate['worker_signature_valid'] = True
            # 2 - validate worker's local_updates_params if worker's signature is valid
            if transaction_to_validate['worker_signature_valid']:
                worker_device_idd = transaction_to_validate["worker_device_idx"]
                # accuracy validated by worker's update
                # accuracy_by_worker_update_using_own_data = self.validate_model_weights(transaction_to_validate["local_updates_params"])
                accuracy_by_worker_update_using_own_data, losses  = self.validate_model_weights1(transaction_to_validate["local_updates_params"], worker_device_idd, comm_round)
                #Compare accuracy with previous average accuracy  OR Compare the pattern of last 3 average accuracies
                file_malicious.write(f"Comm Round: {comm_round}, Previous Average Accuracies:{average_accuracies}, "
                               f"Current Accuracy: {accuracy_by_worker_update_using_own_data} from {worker_device_id} "
                               f"Malicious? {self.devices_dict[worker_device_id].is_malicious}, Losses:{losses}\n")
                # if worker's accuracy larger, or lower but the difference falls within the validator threshold value,
                # meaning worker's updated model favors validator's dataset, so their updates are in the same direction -
                # True, otherwise False. We do not consider the accuracy gap so far, meaning if worker's update is way too good,
                # it is still fine
                print(f'validator updated model accuracy - {self.validator_local_accuracy}')
                print(
                    f"After applying worker's update, model accuracy becomes - {accuracy_by_worker_update_using_own_data}")
                # record their accuracies and difference for choosing a good validator threshold
                is_malicious_validator = "M" if self.is_malicious else "B"
                with open(
                        f"{log_files_folder_path_comm_round}/validator_{self.idx}_{is_malicious_validator}_validation_records_comm_{comm_round}.txt",
                        "a") as file:
                    is_malicious_node = "M" if self.devices_dict[
                        worker_transaction_device_idx].return_is_malicious() else "B"
                    file.write(
                        f"{accuracy_by_worker_update_using_own_data - self.validator_local_accuracy}: validator {self.return_idx()} {is_malicious_validator} in round {comm_round} evluating worker {worker_transaction_device_idx}, diff = v_acc:{self.validator_local_accuracy} - w_acc:{accuracy_by_worker_update_using_own_data} {worker_transaction_device_idx}_maliciousness: {is_malicious_node}\n")

                #DEV just check if the worker was malicious for NOW
                if self.devices_dict[worker_transaction_device_idx].is_malicious:
                # if accuracy_by_worker_update_using_own_data - self.validator_local_accuracy < self.validator_threshold * -1:

                    transaction_to_validate['update_direction'] = False
                    print(
                        f"NOTE: worker {worker_transaction_device_idx}'s updates is deemed as suspiciously malicious by validator {self.idx}")
                    # is it right?
                    if not self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(f"Warning - {worker_transaction_device_idx} is benign and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_negative_malicious_nodes_inside_caught.txt",
                                  'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                else:
                    transaction_to_validate['update_direction'] = True
                    print(
                        f"worker {worker_transaction_device_idx}'s' updates is deemed as GOOD by validator {self.idx}")
                    # is it right?
                    if self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(f"Warning - {worker_transaction_device_idx} is malicious and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt",
                                  'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_positive_good_nodes_inside_correct.txt", 'a') as file:
                            file.write(
                                f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                if self.is_malicious and malicious_validator_on:
                    old_voting = transaction_to_validate['update_direction']
                    transaction_to_validate['update_direction'] = not transaction_to_validate['update_direction']
                    # with open(f"{log_files_folder_path_comm_round}/malicious_validator_log.txt", 'a') as file:
                    #     file.write(
                    #         f"malicious validator {self.idx} has flipped the voting of worker {worker_transaction_device_idx} from {old_voting} to {transaction_to_validate['update_direction']} in round {comm_round}\n")
                transaction_to_validate['validation_rewards'] = 2 * rewards
            else:
                transaction_to_validate['update_direction'] = 'N/A'
                transaction_to_validate['validation_rewards'] = 0
            transaction_to_validate['validation_done_by'] = self.idx
            validation_time = (time.time() - validation_time) / self.computation_power
            transaction_to_validate['validation_time'] = validation_time
            # transaction_to_validate['validator_rsa_pub_key'] = self.return_rsa_pub_key()
            transaction_to_validate['validator_xmss_pub_key'] = self.return_xmss_pub_key()
            self.check_xmss_tree_index()
            transaction_to_validate['validator_tree_no'] = self.return_tree_no()
            # transaction_to_validate['validator_rsa_pub_key'] = self.return_rsa_pub_key()
            # assume signing done in negligible time
            #

            # print(type(transaction_to_validate))
            # df = pd.DataFrame.from_dict(transaction_to_validate)
            # print(df)

            # print(len(transaction_to_validate))
            # print(type(transaction_to_validate))
            # print(transaction_to_validate.keys())

            # print("2. ========VALIDATOR 1==============================",
            #       int.from_bytes(sha256(str(sorted(transaction_to_validate.items())).encode("utf-8")).digest(),
            #                      byteorder='big'))
            # print("7 --------------------------",transaction_to_validate)
            transaction_to_validate["validator_signature"] = self.sign_msg_xmss(sorted(transaction_to_validate.items()))
            # print(". 1000000000000000000 ========VALIDATOR 1 - 1==============================",
            #       int.from_bytes(sha256(str(sorted(transaction_to_validate.items())).encode("utf-8")).digest(),
            #                      byteorder='big'))
            # print(transaction_to_validate["validator_signature"])
            # print("VALIDATOR SIGNING TRANSACTIONS>>>>>>>>>>>>>>>>>>>>>")
            # print("12. ===============BEFORE CONFIRMED=======================",sha256(str(sorted(transaction_to_validate.items())).encode("utf-8")).digest())
            # print(self.xmss_tree.public_key)
            # print("$$$$$$$$$$$$$$$$$$$$", self.xmss_tree.public_key)
            # transaction_to_validate['validator_xmss_pub_key'] = self.return_xmss_pub_key()

            # transaction_to_validate["xmss_tree_validator"] = self.xmss_tree
            return validation_time, transaction_to_validate

class DevicesInNetwork(object):
    def __init__(self, data_set_name, is_iid, batch_size, learning_rate, loss_func, opti, num_devices,
                 network_stability, net, dev, knock_out_rounds, lazy_worker_knock_out_rounds, shard_test_data,
                 miner_acception_wait_time, miner_accepted_transactions_size_limit, validator_threshold, pow_difficulty,
                 even_link_speed_strength, base_data_transmission_speed, even_computation_power, malicious_updates_discount,
                 num_malicious, noise_variance, check_signature, not_resync_chain):
        self.data_set_name = data_set_name
        self.is_iid = is_iid
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.opti = opti
        self.num_devices = num_devices
        self.net = net
        self.dev = dev
        self.devices_set = {}
        self.knock_out_rounds = knock_out_rounds
        self.lazy_worker_knock_out_rounds = lazy_worker_knock_out_rounds
        # self.test_data_loader = None
        self.default_network_stability = network_stability
        self.shard_test_data = shard_test_data
        self.even_link_speed_strength = even_link_speed_strength
        self.base_data_transmission_speed = base_data_transmission_speed
        self.even_computation_power = even_computation_power
        self.num_malicious = num_malicious
        self.malicious_updates_discount = malicious_updates_discount
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        self.not_resync_chain = not_resync_chain
        # distribute dataset
        ''' validator '''
        self.validator_threshold = validator_threshold
        ''' miner '''
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        self.pow_difficulty = pow_difficulty
        ''' shard '''
        self.data_set_balanced_allocation()

    # distribute the dataset evenly to the devices
    def data_set_balanced_allocation(self):
        # read dataset
        mnist_dataset = DatasetLoad(self.data_set_name, self.is_iid)

        # perpare training data
        train_data = mnist_dataset.train_data
        train_label = mnist_dataset.train_label
        # shard dataset and distribute among devices
        # shard train
        shard_size_train = mnist_dataset.train_data_size // self.num_devices // 2
        shards_id_train = np.random.permutation(mnist_dataset.train_data_size // shard_size_train)

        # perpare test data
        if not self.shard_test_data:
            test_data = torch.tensor(mnist_dataset.test_data)
            test_label = torch.argmax(torch.tensor(mnist_dataset.test_label), dim=1)
            test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
        else:
            test_data = mnist_dataset.test_data
            test_label = mnist_dataset.test_label
             # shard test
            shard_size_test = mnist_dataset.test_data_size // self.num_devices // 2
            shards_id_test = np.random.permutation(mnist_dataset.test_data_size // shard_size_test)

        # malicious_nodes_set = []
        # if self.num_malicious:
        #     malicious_nodes_set = random.sample(range(self.num_devices), self.num_malicious)



        for i in range(self.num_devices):
            is_malicious = False
            # make it more random by introducing two shards
            shards_id_train1 = shards_id_train[i * 2]
            shards_id_train2 = shards_id_train[i * 2 + 1]
            # distribute training data
            data_shards1 = train_data[shards_id_train1 * shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]
            data_shards2 = train_data[shards_id_train2 * shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            label_shards1 = train_label[shards_id_train1 * shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]
            label_shards2 = train_label[shards_id_train2 * shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            local_train_data, local_train_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_train_label = np.argmax(local_train_label, axis=1)
            # distribute test data
            if self.shard_test_data:
                shards_id_test1 = shards_id_test[i * 2]
                shards_id_test2 = shards_id_test[i * 2 + 1]
                data_shards1 = test_data[shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                data_shards2 = test_data[shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                label_shards1 = test_label[shards_id_test1 * shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                label_shards2 = test_label[shards_id_test2 * shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                local_test_data, local_test_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
                local_test_label = torch.argmax(torch.tensor(local_test_label), dim=1)
                test_data_loader = DataLoader(TensorDataset(torch.tensor(local_test_data), torch.tensor(local_test_label)), batch_size=100, shuffle=False)
            # assign data to a device and put in the devices set
            # if i in malicious_nodes_set:
            #     is_malicious = True
                # add Gussian Noise

            device_idx = f'device_{i+1}'
            a_device = Device(device_idx, TensorDataset(torch.tensor(local_train_data),
                                                        torch.tensor(local_train_label)), test_data_loader,
                              self.batch_size, self.learning_rate, self.loss_func,
                              self.opti, self.default_network_stability, self.net, self.dev,
                              self.miner_acception_wait_time, self.miner_accepted_transactions_size_limit,
                              self.validator_threshold, self.pow_difficulty, self.even_link_speed_strength,
                              self.base_data_transmission_speed, self.even_computation_power, is_malicious,
                              self.noise_variance, self.check_signature, self.not_resync_chain,
                              self.malicious_updates_discount, self.knock_out_rounds, self.lazy_worker_knock_out_rounds)
            # device index starts from 1
            self.devices_set[device_idx] = a_device
            print(f"Sharding dataset to {device_idx} done.")
        print(f"Sharding dataset done!")