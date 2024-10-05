import os
import sys
import argparse
import random
import time
from datetime import datetime
import copy
from sys import getsizeof
import sqlite3
import pickle
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from Models import Mnist_2NN, Mnist_CNN
from Device import DevicesInNetwork

from sklearn.preprocessing import normalize

import logging

logging.basicConfig(filename="log.txt", level=logging.DEBUG)

date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
log_files_folder_path = f"logs/{date_time}"
NETWORK_SNAPSHOTS_BASE_FOLDER = "snapshots"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Block_FedAvg_Simulation")

parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-v', '--verbose', type=int, default=1, help='print verbose debug log')
parser.add_argument('-sn', '--save_network_snapshots', type=int, default=0,
                    help='only save network_snapshots if this is set to 1; will create a folder with date in the snapshots folder')
parser.add_argument('-dtx', '--destroy_tx_in_block', type=int, default=0,
                    help='currently transactions stored in the blocks are occupying GPU ram and have not figured out a way to move them to CPU ram or harddisk, so turn it on to save GPU ram in order for PoS to run 100+ rounds. NOT GOOD if there needs to perform chain resyncing.')
parser.add_argument('-rp', '--resume_path', type=str, default=None,
                    help='resume from the path of saved network_snapshots; only provide the date')
parser.add_argument('-sf', '--save_freq', type=int, default=5, help='save frequency of the network_snapshot')
parser.add_argument('-sm', '--save_most_recent', type=int, default=2,
                    help='in case of saving space, keep only the recent specified number of snapshops; 0 means keep all')

parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01,
                    help="learning rate, use value from origin paper as default")
parser.add_argument('-op', '--optimizer', type=str, default="SGD",
                    help='optimizer to be used, by default implementing stochastic gradient descent')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to devices')
parser.add_argument('-max_ncomm', '--max_num_comm', type=int, default=100,
                    help='maximum number of communication rounds, may terminate early if converges')
parser.add_argument('-nd', '--num_devices', type=int, default=20, help='numer of the devices in the simulation network')
parser.add_argument('-st', '--shard_test_data', type=int, default=0,
                    help='it is easy to see the global models are consistent across devices when the test dataset is NOT sharded')
parser.add_argument('-nm', '--num_malicious', type=int, default=0,
                    help="number of malicious nodes in the network. malicious node's data sets will be introduced Gaussian noise")
parser.add_argument('-nv', '--noise_variance', type=int, default=1,
                    help="noise variance level of the injected Gaussian Noise")
parser.add_argument('-le', '--default_local_epochs', type=int, default=5,
                    help='local train epoch. Train local model by this same num of epochs for each worker, if -mt is not specified')
parser.add_argument('-ur', '--unit_reward', type=int, default=1,
                    help='unit reward for providing data, verification of signature, validation and so forth')
parser.add_argument('-ko', '--knock_out_rounds', type=int, default=6,
                    help="a worker or validator device is kicked out of the device's peer list(put in black list) if it's identified as malicious for this number of rounds")
parser.add_argument('-lo', '--lazy_worker_knock_out_rounds', type=int, default=10,
                    help="a worker device is kicked out of the device's peer list(put in black list) if it does not provide updates for this number of rounds, due to too slow or just lazy to do updates and only accept the model udpates.(do not care lazy validator or miner as they will just not receive rewards)")
parser.add_argument('-pow', '--pow_difficulty', type=int, default=0, help="if set to 0, meaning miners are using PoS")
parser.add_argument('-mt', '--miner_acception_wait_time', type=float, default=0.0,
                    help="default time window for miners to accept transactions, in seconds. 0 means no time limit, and each device will just perform same amount(-le) of epochs per round like in FedAvg paper")
parser.add_argument('-ml', '--miner_accepted_transactions_size_limit', type=float, default=0.0,
                    help="no further transactions will be accepted by miner after this limit. 0 means no size limit. either this or -mt has to be specified, or both. This param determines the final block_size")
parser.add_argument('-mp', '--miner_pos_propagated_block_wait_time', type=float, default=float("inf"),
                    help="this wait time is counted from the beginning of the comm round, used to simulate forking events in PoS")
parser.add_argument('-vh', '--validator_threshold', type=float, default=1.0,
                    help="a threshold value of accuracy difference to determine malicious worker")
parser.add_argument('-md', '--malicious_updates_discount', type=float, default=0.0,
                    help="do not entirely drop the voted negative worker transaction because that risks the same worker dropping the entire transactions and repeat its accuracy again and again and will be kicked out. Apply a discount factor instead to the false negative worker's updates are by some rate applied so it won't repeat")
parser.add_argument('-mv', '--malicious_validator_on', type=int, default=0,
                    help="let malicious validator flip voting result")
parser.add_argument('-ns', '--network_stability', type=float, default=1.0, help='the odds a device is online')
parser.add_argument('-els', '--even_link_speed_strength', type=int, default=1,
                    help="This variable is used to simulate transmission delay. "
                         "Default value 1 means every device is assigned to the same link speed strength -dts bytes/sec. If set to 0, link speed strength is randomly initiated between 0 and 1, meaning a device will transmit  -els*-dts bytes/sec - during experiment, one transaction is around 35k bytes.")
parser.add_argument('-dts', '--base_data_transmission_speed', type=float, default=70000.0,
                    help="volume of data can be transmitted per second when -els == 1. set this variable to determine transmission speed (bandwidth), which further determines the transmission delay - during experiment, one transaction is around 35k bytes.")
parser.add_argument('-ecp', '--even_computation_power', type=int, default=1,
                    help="This variable is used to simulate strength of hardware equipment. The calculation time will be shrunk down by this value. Default value 1 means evenly assign computation power to 1. If set to 0, power is randomly initiated as an int between 0 and 4, both included.")

parser.add_argument('-ha', '--hard_assign', type=str, default='*,*,*',
                    help="hard assign number of roles in the network, order by worker, validator and miner. e.g. 12,5,3 assign 12 workers, 5 validators and 3 miners. \"*,*,*\" means completely random role-assigning in each communication round ")
parser.add_argument('-aio', '--all_in_one', type=int, default=1,
                    help='let all nodes be aware of each other in the network while registering')
parser.add_argument('-cs', '--check_signature', type=int, default=1,
                    help='if set to 0, all signatures are assumed to be verified to save execution time')

dict = {
    "comm_round": [],
    "stake_list": [],
    "average": []

}

dict1 = {
    "comm_round": [],
    "stake_list": [],
    "average": []

}

dict_devices = {
    "device": [],
    "stake": [],
    "value": []
}

dict_devices1 = {}

final_dict = {
    "device": [],
    "stake": [],
    "computation": [],
    "role_selected": [],
    "selection_value": [],
    "vrf_output": [],
    "contribution": []
}

off_on_devices = {
    "comm_round": [],
    "online": [],
    "offline": []
}

from hashlib import sha256
from struct import unpack

import vrf_helpers_central


def vrf(alphastring):
    value = []

    vrf_sk = str(random.seed(10)).encode("utf-8")
    vrf_pk = vrf_helpers_central.get_public_key(vrf_sk)
    p_status, pi_string = vrf_helpers_central.ecvrf_prove(vrf_sk, alphastring)

    b_status, beta_string = vrf_helpers_central.ecvrf_proof_to_hash(pi_string)
    beta_sum = sum(list(beta_string))
    result, beta_string2 = vrf_helpers_central.ecvrf_verify(vrf_pk, pi_string, alphastring)
    if p_status == "VALID" and \
            b_status == "VALID" and \
            result == "VALID" and \
            beta_string == beta_string2:
        return True, beta_string
    return False, b'null'


def bytes_to_float(byte_value):
    return float(unpack('L', sha256(byte_value).digest()[:8])[0]) / 2 ** (len(byte_value))


average_accuracies = []

if __name__ == "__main__":

    if not os.path.exists('logs'):
        os.makedirs('logs')

    if not os.path.exists('benchmark'):
        os.makedirs('benchmark')

    bench_folder = f"benchmark"

    args = parser.parse_args()
    args = args.__dict__

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    latest_round_num = 0

    ''' If network_snapshot is specified, continue from left '''
    if args['resume_path']:
        if not args['save_network_snapshots']:
            print("NOTE: save_network_snapshots is set to 0. New network_snapshots won't be saved by conituing.")
        network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{args['resume_path']}"
        latest_network_snapshot_file_name = \
        sorted([f for f in os.listdir(network_snapshot_save_path) if not f.startswith('.')],
               key=lambda fn: int(fn.split('_')[-1]), reverse=True)[0]
        print(f"Loading network snapshot from {args['resume_path']}/{latest_network_snapshot_file_name}")
        print("BE CAREFUL - loaded dev env must be the same as the current dev env, namely, cpu, gpu or gpu parallel")
        latest_round_num = int(latest_network_snapshot_file_name.split('_')[-1])
        devices_in_network = pickle.load(
            open(f"{network_snapshot_save_path}/{latest_network_snapshot_file_name}", "rb"))
        devices_list = list(devices_in_network.devices_set.values())
        log_files_folder_path = f"logs/{args['resume_path']}"

        args_used_file = f"{log_files_folder_path}/args_used.txt"
        file = open(args_used_file, "r")
        log_whole_text = file.read()
        lines_list = log_whole_text.split("\n")
        for line in lines_list:

            if line.startswith('--unit_reward'):
                rewards = int(line.split(" ")[-1])

            if line.startswith('--hard_assign'):
                roles_requirement = line.split(" ")[-1].split(',')

            if line.startswith('--pow_difficulty'):
                mining_consensus = 'PoW' if int(line.split(" ")[-1]) else 'PoS'

        try:
            workers_needed = int(roles_requirement[0])
        except:
            workers_needed = 1
        try:
            validators_needed = int(roles_requirement[1])
        except:
            validators_needed = 1
        try:
            miners_needed = int(roles_requirement[2])
        except:
            miners_needed = 1

    else:
        ''' SETTING UP FROM SCRATCH'''

        os.mkdir(log_files_folder_path)

        with open(f'{log_files_folder_path}/args_used.txt', 'w') as f:
            f.write("Command line arguments used -\n")
            f.write(' '.join(sys.argv[1:]))
            f.write("\n\nAll arguments used -\n")
            for arg_name, arg in args.items():
                f.write(f'\n--{arg_name} {arg}')

        if args['save_network_snapshots']:
            network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{date_time}"
            os.mkdir(network_snapshot_save_path)

        rewards = args["unit_reward"]

        roles_requirement = args['hard_assign'].split(',')

        try:
            workers_needed = int(roles_requirement[0])
        except:
            workers_needed = 1
        try:
            validators_needed = int(roles_requirement[1])
        except:
            validators_needed = 1
        try:
            miners_needed = int(roles_requirement[2])
        except:
            miners_needed = 1

        num_devices = args['num_devices']
        num_malicious = args['num_malicious']

        if num_devices < workers_needed + miners_needed + validators_needed:
            sys.exit(
                "ERROR: Roles assigned to the devices exceed the maximum number of allowed devices in the network.")

        if num_devices < 3:
            sys.exit(
                "ERROR: There are not enough devices in the network.\n The system needs at least one miner, one worker and/or one validator to start the operation.\nSystem aborted.")

        if num_malicious:
            if num_malicious > num_devices:
                sys.exit(
                    "ERROR: The number of malicious nodes cannot exceed the total number of devices set in this network")
            else:
                print(
                    f"Malicious nodes vs total devices set to {num_malicious}/{num_devices} = {(num_malicious / num_devices) * 100:.2f}%")

        neural_net = None
        if args['model_name'] == 'mnist_2nn':
            neural_net = Mnist_2NN()
        elif args['model_name'] == 'mnist_cnn':
            neural_net = Mnist_CNN()

        if torch.cuda.device_count() > 1:
            neural_net = torch.nn.DataParallel(neural_net)
        print(f"{torch.cuda.device_count()} GPUs are available to use!")
        neural_net = neural_net.to(dev)

        loss_func = F.cross_entropy

        devices_in_network = DevicesInNetwork(data_set_name='mnist', is_iid=args['IID'],
                                              batch_size=args['batchsize'],
                                              learning_rate=args['learning_rate'], loss_func=loss_func,
                                              opti=args['optimizer'], num_devices=num_devices,
                                              network_stability=args['network_stability'], net=neural_net, dev=dev,
                                              knock_out_rounds=args['knock_out_rounds'],
                                              lazy_worker_knock_out_rounds=args['lazy_worker_knock_out_rounds'],
                                              shard_test_data=args['shard_test_data'],
                                              miner_acception_wait_time=args['miner_acception_wait_time'],
                                              miner_accepted_transactions_size_limit=args[
                                                  'miner_accepted_transactions_size_limit'],
                                              validator_threshold=args['validator_threshold'],
                                              pow_difficulty=args['pow_difficulty'],
                                              even_link_speed_strength=args['even_link_speed_strength'],
                                              base_data_transmission_speed=args['base_data_transmission_speed'],
                                              even_computation_power=args['even_computation_power'],
                                              malicious_updates_discount=args['malicious_updates_discount'],
                                              num_malicious=num_malicious, noise_variance=args['noise_variance'],
                                              check_signature=args['check_signature'],
                                              not_resync_chain=args['destroy_tx_in_block'])
        del neural_net
        devices_list = list(devices_in_network.devices_set.values())

        for device in devices_list:
            device.init_global_parameters()

            device.set_devices_dict_and_aio(devices_in_network.devices_set, args["all_in_one"])

            device.register_in_the_network()

        for device in devices_list:
            device.remove_peers(device)

        mining_consensus = 'PoW' if args['pow_difficulty'] else 'PoS'

    conn = sqlite3.connect(f'{log_files_folder_path}/malicious_wokrer_identifying_log.db')
    conn_cursor = conn.cursor()
    conn_cursor.execute("""CREATE TABLE if not exists  malicious_workers_log (
    device_seq text,
    if_malicious integer,
    correctly_identified_by text,
    incorrectly_identified_by text,
    in_round integer,
    when_resyncing text
    )""")

    for comm_round in range(latest_round_num + 1, args['max_num_comm'] + 1):

        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if os.path.exists(log_files_folder_path_comm_round):
            print(f"Deleting {log_files_folder_path_comm_round} and create a new one.")
            shutil.rmtree(log_files_folder_path_comm_round)
        os.mkdir(log_files_folder_path_comm_round)

        if dev == torch.device("cuda"):
            with torch.cuda.device('cuda'):
                torch.cuda.empty_cache()

        comm_round_start_time = time.time()

        workers_to_assign = workers_needed
        miners_to_assign = miners_needed
        validators_to_assign = validators_needed
        workers_this_round = []
        miners_this_round = []
        validators_this_round = []

        random.shuffle(devices_list)

        num_of_offline = 0

        stake_list_clip = []
        stake_list = []
        stake_total = 0
        stake_list_clip = []
        for device in devices_list:
            stake = device.return_stake()
            stake_list.append(stake)
            stake_total += int(stake)

        stake_average = stake_total / len(stake_list)

        a = stake_list

        normal_list = normalize([a])

        lt = normal_list[0].tolist()
        stake_list1 = lt

        normal_avg = sum(stake_list1) / len(stake_list1)

        stakefile = open(f"{bench_folder}/file1.txt", "a")
        if comm_round % 10 == 0:
            stakefile.write(
                f"COMM_ROUND: {comm_round} | Normalized: {stake_list1} | Actual: {stake_list} | Average: {stake_average} | AVG: {normal_avg}\n")

        dict["comm_round"].append(comm_round)
        dict["stake_list"].append(stake_list)
        dict["average"].append(stake_average)
        filedict = open(f"{bench_folder}/file2.json", "a")
        filedict.write(str(dict))

        dict1["comm_round"].append(comm_round)
        dict1["stake_list"].append(stake_list1)
        dict1["average"].append(normal_avg)

        filedict = open(f"{bench_folder}/file3.json", "a")
        if comm_round % 5 == 0:
            filedict.write(str(dict1))

        selection_list = []
        devicesss = []
        valuesss = []

        for i, device in enumerate(devices_list):
            device.is_malicious = False
            value_encoded = str(device.blockchain.return_last_block()).encode()

            result, beta_string = device.vrf(value_encoded)
            ratio = device.bytes_to_float(beta_string)

            p_role = 1 / 3
            p_ratio = ratio
            p_stake = stake_list1[i]

            c_power = random.randint(1, 3)

            contribution_value = device.return_contribution_value()
            a = 1
            b = 1
            c = 1
            d = 1
            e = 1

            final_value = (b * p_ratio) + (c * p_stake) + (e * c_power) + contribution_value

            device.set_computation_power(c_power)
            device.set_selection_value(final_value)
            device.set_vrf_output(p_ratio)

        devices_list.sort(key=lambda device: device.return_selection_value())

        total_devices = len(devices_list)

        sum_of_ratio = 5 + 2 + 1
        w_ratio = int((5 / sum_of_ratio) * total_devices)
        v_ratio = int((2 / sum_of_ratio) * total_devices)
        m_ratio = int((1 / sum_of_ratio) * total_devices)

        file_record = open(f"{bench_folder}/roleplot.txt", "a")
        msg = ""
        for i, device in enumerate(devices_list):
            p_power = device.return_computation_power()

            p_stake = device.return_stake()
            p_ratio = device.return_vrf_output()

            final_value = device.return_selection_value()
            contribution_value = device.return_contribution_value()

            final_dict["device"].append(device.return_idx())
            final_dict["computation"].append(p_power)
            final_dict["stake"].append(p_stake)
            final_dict["vrf_output"].append(p_ratio)
            final_dict["selection_value"].append(final_value)
            final_dict["contribution"].append(contribution_value)

            if device.is_online():
                if i < w_ratio:

                    device.role = "worker"
                    role_selected = "worker"
                    msg = f"COMM_ROUND: {comm_round}| {device.idx} with stake: {device.return_stake()} selected WORKER | Selection Value: {final_value}"

                elif i >= w_ratio and i < (w_ratio + v_ratio):

                    device.role = "validator"
                    role_selected = "validator"
                    msg = f"COMM_ROUND: {comm_round}| {device.idx} with stake: {device.return_stake()} selected VALIDATOR  | Selection Value: {final_value}"
                else:

                    device.role = "miner"
                    role_selected = "miner"
                    msg = f"COMM_ROUND: {comm_round}| {device.idx} with stake: {device.return_stake()} selected MINER  | Selection Value: {final_value}"

                if comm_round % 25 == 0:
                    file_record.write(msg + "\n")

            else:
                role_selected = "offline"
                device.role = "offline"

            final_dict["role_selected"].append(role_selected)
            if comm_round % 10 == 0:
                file_record.write(f"{final_dict}\n")

        for device in devices_list:
            if device.is_online():
                if device.return_role() == 'worker':

                    workers_this_round.append(device)

                elif device.return_role() == 'miner':
                    miners_this_round.append(device)

                else:
                    validators_this_round.append(device)

        if args['verbose']:

            for device in devices_list:
                if device.is_online():
                    print(f'{device.return_idx()} {device.return_role()} online - ', end='')
                else:
                    print(f'{device.return_idx()} {device.return_role()} offline - ', end='')

                print(f"chain length {device.return_blockchain_object().return_chain_length()}")

            print(
                f"\nThere are {len(workers_this_round)} workers, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round.")
            print("\nworkers this round are")
            for worker in workers_this_round:
                print(
                    f"d_{worker.return_idx().split('_')[-1]} online - {worker.is_online()} with chain len {worker.return_blockchain_object().return_chain_length()}")
            print("\nminers this round are")
            for miner in miners_this_round:
                print(
                    f"d_{miner.return_idx().split('_')[-1]} online - {miner.is_online()} with chain len {miner.return_blockchain_object().return_chain_length()}")
            print("\nvalidators this round are")
            for validator in validators_this_round:
                print(
                    f"d_{validator.return_idx().split('_')[-1]} online - {validator.is_online()} with chain len {validator.return_blockchain_object().return_chain_length()}")
            print()

            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")
            for device_seq, device in devices_in_network.devices_set.items():

                peers = device.return_peers()

                print(f"d_{device_seq.split('_')[-1]} - {device.return_role()[0]} has peer list ", end='')
                for peer in peers:
                    print(f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()

            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")

        for miner in miners_this_round:
            miner.miner_reset_vars_for_new_round()
        for worker in workers_this_round:
            worker.worker_reset_vars_for_new_round()
        for validator in validators_this_round:
            validator.validator_reset_vars_for_new_round()

        random.shuffle(workers_this_round)
        random.shuffle(miners_this_round)
        random.shuffle(validators_this_round)

        ''' workers, validators and miners take turns to perform jobs '''

        print(
            ''' Step 1 - workers assign associated miner and validator (and do local updates, but it is implemented in code block of step 2) \n''')
        for worker_iter in range(len(workers_this_round)):
            worker = workers_this_round[worker_iter]

            if worker.resync_chain(mining_consensus):
                worker.update_model_after_chain_resync(log_files_folder_path_comm_round, conn, conn_cursor)

            print(
                f"{worker.return_idx()} - worker {worker_iter + 1}/{len(workers_this_round)} will associate with a validator and a miner, if online...")

            if worker.is_online():
                associated_miner = worker.associate_with_device("miner")
                if associated_miner:
                    associated_miner.add_device_to_association(worker)
                else:
                    print(f"Cannot find a qualified miner in {worker.return_idx()} peer list.")

            if worker.is_online():

                associated_validator = worker.associate_with_device("validator")
                if associated_validator:
                    associated_validator.add_device_to_association(worker)
                else:
                    print(f"Cannot find a qualified validator in {worker.return_idx()} peer list.")

        print(
            ''' Step 2 - validators accept local updates and broadcast to other validators in their respective peer lists (workers local_updates() are called in this step.\n''')
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]

            if validator.resync_chain(mining_consensus):
                validator.update_model_after_chain_resync(log_files_folder_path, conn, conn_cursor)

            if validator.is_online():
                associated_miner = validator.associate_with_device("miner")
                if associated_miner:
                    associated_miner.add_device_to_association(validator)
                else:
                    print(f"Cannot find a qualified miner in validator {validator.return_idx()} peer list.")

            associated_workers = list(validator.return_associated_workers())
            if not associated_workers:
                print(
                    f"No workers are associated with validator {validator.return_idx()} {validator_iter + 1}/{len(validators_this_round)} for this communication round.")
                continue
            validator_link_speed = validator.return_link_speed()
            print(
                f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} is accepting workers' updates with link speed {validator_link_speed} bytes/s, if online...")

            records_dict = dict.fromkeys(associated_workers, None)
            for worker, _ in records_dict.items():
                records_dict[worker] = {}

            transaction_arrival_queue = {}

            if args['miner_acception_wait_time']:
                print(
                    f"miner wati time is specified as {args['miner_acception_wait_time']} seconds. let each worker do local_updates till time limit")
                for worker_iter in range(len(associated_workers)):
                    worker = associated_workers[worker_iter]
                    if not worker.return_idx() in validator.return_black_list():

                        print(
                            f'worker {worker_iter + 1}/{len(associated_workers)} of validator {validator.return_idx()} is doing local updates')
                        total_time_tracker = 0
                        update_iter = 1
                        worker_link_speed = worker.return_link_speed()
                        lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
                        while total_time_tracker < validator.return_miner_acception_wait_time():

                            if worker.is_online():
                                local_update_spent_time = worker.worker_local_update(rewards,
                                                                                     log_files_folder_path_comm_round,
                                                                                     comm_round)
                                unverified_transaction = worker.return_local_updates_and_signature(comm_round)

                                unverified_transactions_size = getsizeof(str(unverified_transaction))

                                if local_update_spent_time + transmission_delay > validator.return_miner_acception_wait_time():
                                    break
                                records_dict[worker][update_iter] = {}
                                records_dict[worker][update_iter]['local_update_time'] = local_update_spent_time
                                records_dict[worker][update_iter]['transmission_delay'] = transmission_delay
                                records_dict[worker][update_iter][
                                    'local_update_unverified_transaction'] = unverified_transaction
                                records_dict[worker][update_iter][
                                    'local_update_unverified_transaction_size'] = unverified_transactions_size
                                if update_iter == 1:
                                    total_time_tracker = local_update_spent_time + transmission_delay
                                else:
                                    total_time_tracker = total_time_tracker - records_dict[worker][update_iter - 1][
                                        'transmission_delay'] + local_update_spent_time + transmission_delay
                                records_dict[worker][update_iter]['arrival_time'] = total_time_tracker
                                if validator.is_online():

                                    print(f"validator {validator.return_idx()} has accepted this transaction.")
                                    transaction_arrival_queue[total_time_tracker] = unverified_transaction
                                else:
                                    print(
                                        f"validator {validator.return_idx()} offline and unable to accept this transaction")
                            else:

                                wasted_update_time, wasted_update_params = worker.waste_one_epoch_local_update_time(
                                    args['optimizer'])
                                wasted_update_params_size = getsizeof(str(wasted_update_params))
                                wasted_transmission_delay = wasted_update_params_size / lower_link_speed
                                if wasted_update_time + wasted_transmission_delay > validator.return_miner_acception_wait_time():
                                    break
                                records_dict[worker][update_iter] = {}
                                records_dict[worker][update_iter]['transmission_delay'] = transmission_delay
                                if update_iter == 1:
                                    total_time_tracker = wasted_update_time + wasted_transmission_delay
                                    print(
                                        f"worker goes offline and wasted {total_time_tracker} seconds for a transaction")
                                else:
                                    total_time_tracker = total_time_tracker - records_dict[worker][update_iter - 1][
                                        'transmission_delay'] + wasted_update_time + wasted_transmission_delay
                            update_iter += 1
            else:

                for worker_iter in range(len(associated_workers)):
                    worker = associated_workers[worker_iter]
                    if not worker.return_idx() in validator.return_black_list():
                        print(
                            f'worker {worker_iter + 1}/{len(associated_workers)} of validator {validator.return_idx()} is doing local updates')
                        if worker.is_online():
                            local_update_spent_time = worker.worker_local_update(rewards,
                                                                                 log_files_folder_path_comm_round,
                                                                                 comm_round,
                                                                                 local_epochs=args[
                                                                                     'default_local_epochs'])
                            worker_link_speed = worker.return_link_speed()
                            lower_link_speed = validator_link_speed if validator_link_speed < worker_link_speed else worker_link_speed
                            unverified_transaction = worker.return_local_updates_and_signature(comm_round)

                            unverified_transactions_size = getsizeof(str(unverified_transaction))
                            transmission_delay = unverified_transactions_size / lower_link_speed

                            with open(f"{bench_folder}/unverified_transaction_size.txt", "a") as file:
                                file.write(
                                    f"{comm_round}:{worker.return_idx()} - Unverified Transaction SIZE: {unverified_transactions_size}\n")

                            transmission_delay = unverified_transactions_size / lower_link_speed
                            with open(f"{bench_folder}/transmission_delay", "a") as file:
                                file.write(f"Transmission Delay:{transmission_delay}\n")

                            if validator.is_online():
                                transaction_arrival_queue[
                                    local_update_spent_time + transmission_delay] = unverified_transaction
                                print(f"validator {validator.return_idx()} has accepted this transaction.")
                            else:
                                print(
                                    f"validator {validator.return_idx()} offline and unable to accept this transaction")
                        else:
                            print(f"worker {worker.return_idx()} offline and unable do local updates")
                    else:
                        print(
                            f"worker {worker.return_idx()} in validator {validator.return_idx()}'s black list. This worker's transactions won't be accpeted.")
            validator.set_unordered_arrival_time_accepted_worker_transactions(transaction_arrival_queue)

            validator.set_transaction_for_final_validating_queue(sorted(transaction_arrival_queue.items()))

            if transaction_arrival_queue:
                validator.validator_broadcast_worker_transactions()
            else:
                print(
                    "No transactions have been received by this validator, probably due to workers and/or validators offline or timeout while doing local updates or transmitting updates, or all workers are in validator's black list.")

        print(
            ''' Step 2.5 - with the broadcasted workers transactions, validators decide the final transaction arrival order \n''')

        print(
            ''' Step 3 - validators do self and cross-validation(validate local updates from workers) by the order of transaction arrival time.\n''')

        all_losses = []
        all_accuracies = []
        for validator_iter in range(len(validators_this_round)):
            validator = validators_this_round[validator_iter]
            final_transactions_arrival_queue = validator.return_final_transactions_validating_queue()
            if final_transactions_arrival_queue:

                local_validation_time = validator.validator_update_model_by_one_epoch_and_validate_local_accuracy(
                    args['optimizer'])
                print(
                    f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} is validating received worker transactions...")
                validation_time = time.time()

                for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
                    if arrival_time < local_validation_time:
                        arrival_time = local_validation_time
                    worker_transaction_device_idx = unconfirmmed_transaction['worker_device_idx']
                    if worker_transaction_device_idx in validator.black_list:
                        print(
                            f"{worker_transaction_device_idx} is in validator's blacklist. Trasaction won't get validated.")

                    if validator.check_signature:
                        transaction_before_signed = copy.deepcopy(unconfirmmed_transaction)
                        del transaction_before_signed["worker_signature"]

                        pub_key = unconfirmmed_transaction['worker_xmss_pub_key']["pub_key"]
                        signature = unconfirmmed_transaction["worker_signature"]
                        worker_tree_no = unconfirmmed_transaction["worker_tree_no"]
                        worker_device_id = unconfirmmed_transaction["worker_device_idx"]

                        hash = int.from_bytes(
                            sha256(str(sorted(transaction_before_signed.items())).encode('utf-8')).digest(),
                            byteorder='big')

                        if validator.verify_msg_xmss(hash, signature, pub_key) and validator.verify_dilithium(pub_key,
                                                                                                              worker_device_id,
                                                                                                              worker_tree_no):
                            print(
                                f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {validator.idx}!")
                            unconfirmmed_transaction['worker_signature_valid'] = True
                        else:
                            print(
                                f"Signature invalid. Transaction from worker {worker_transaction_device_idx} does NOT pass verification.")

                            unconfirmmed_transaction['worker_signature_valid'] = False
                    else:
                        print(
                            f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {validator.idx}!")
                        unconfirmmed_transaction['worker_signature_valid'] = True

                total_losses = 0
                total_accuracy = 0
                average_loss = 0
                counter = 0

                lossesArray = []
                lossOnly = []
                workers_ids = []

                for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:
                    counter += 1
                    worker_device_idd = unconfirmmed_transaction["worker_device_idx"]
                    workers_ids.append(worker_device_idd)

                    accuracy_by_worker_update_using_own_data, losses = validator.validate_model_weights1(
                        unconfirmmed_transaction["local_updates_params"], worker_device_idd, comm_round)
                    all_losses.append(sum(losses) / len(losses))
                    all_accuracies.append(accuracy_by_worker_update_using_own_data)
                    lossesArray.append(
                        f"{sum(losses) / len(losses)}: Worker: {worker_device_idd} by Validator: {validator.idx}")
                    validator.devices_dict[worker_device_idd].validation_loss = sum(losses) / len(losses)
                    validator.devices_dict[
                        worker_device_idd].validation_accuracy = accuracy_by_worker_update_using_own_data
                    total_losses += sum(losses) / len(losses)
                    total_accuracy += accuracy_by_worker_update_using_own_data

                average_loss = total_losses / counter
                average_accuracy = total_accuracy / counter

                with open(f"{bench_folder}/loss.txt", "a") as file:
                    file.write(f"Communication Round: {comm_round} - Loss Array: {lossesArray}\n")
                with open(f"{bench_folder}/all_losses.txt", "a") as file:
                    file.write(f"Communication Round: {comm_round} - All Loss Array: {all_losses}\n")
                with open(f"{bench_folder}/all_accuracies.txt", "a") as file:
                    file.write(f"Communication Round: {comm_round} - All Accuracies: {all_accuracies}\n")

                for (arrival_time, unconfirmmed_transaction) in final_transactions_arrival_queue:

                    if unconfirmmed_transaction['worker_signature_valid']:
                        worker_device_idd = unconfirmmed_transaction["worker_device_idx"]

                        if validator.devices_dict[worker_device_idd].return_is_malicious():

                            validator.devices_dict[worker_transaction_device_idx].detectedMalicious = True
                            unconfirmmed_transaction['update_direction'] = False
                        else:
                            unconfirmmed_transaction['update_direction'] = True
                            print(
                                f"worker {worker_transaction_device_idx}'s' updates is deemed as GOOD by validator {validator.idx}")

                        unconfirmmed_transaction['validation_rewards'] = 2 * rewards
                    else:
                        unconfirmmed_transaction['update_direction'] = 'N/A'
                        unconfirmmed_transaction['validation_rewards'] = 0
                    unconfirmmed_transaction['validation_done_by'] = validator.idx
                    validation_time = (time.time() - validation_time) / validator.computation_power
                    unconfirmmed_transaction['validation_time'] = validation_time

                    unconfirmmed_transaction['validator_xmss_pub_key'] = validator.return_xmss_pub_key()
                    validator.check_xmss_tree_index()
                    unconfirmmed_transaction['validator_tree_no'] = validator.return_tree_no()

                    unconfirmmed_transaction["validator_signature"] = validator.sign_msg_xmss(
                        sorted(unconfirmmed_transaction.items()))

                    if validation_time:
                        validator.add_post_validation_transaction_to_queue((arrival_time + validation_time,
                                                                            validator.return_link_speed(),
                                                                            unconfirmmed_transaction))
                        print(
                            f"A validation process has been done for the transaction from worker {unconfirmmed_transaction['worker_device_idx']} by validator {validator.return_idx()}")
            else:
                print(
                    f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} did not receive any transaction from worker or validator in this round.")

        print(
            ''' Step 6 last step - process the added block - 1.collect usable updated params\n 2.malicious nodes identification\n 3.get rewards\n 4.do local udpates\n This code block is skipped if no valid block was generated in this round''')
        all_devices_round_ends_time = []
        local_params_by_benign_workers = []
        for device in devices_list:
            if device.role == "worker":
                local_params_by_benign_workers.append(device.local_train_parameters)
        for device in devices_list:

            processing_time = time.time()

            sum_parameters = None
            for local_updates_params in local_params_by_benign_workers:
                if sum_parameters is None:
                    sum_parameters = copy.deepcopy(local_updates_params)
                else:
                    for var in sum_parameters:
                        sum_parameters[var] += local_updates_params[var]

            num_participants = len(local_params_by_benign_workers)
            for var in device.global_parameters:
                device.global_parameters[var] = (sum_parameters[var] / num_participants)
            print(f"global updates done by {device.idx}")

            processing_time = (time.time() - processing_time) / device.computation_power
            device.other_tasks_at_the_end_of_comm_round(comm_round, log_files_folder_path)
            device.add_to_round_end_time(processing_time)
            all_devices_round_ends_time.append(device.return_round_end_time())

        print(''' Logging Accuracies by Devices ''')

        average_accuracy = 0
        total_accuracy = 0
        total_honest_online_devices = 0
        for device in devices_list:

            if not device.is_malicious and device.is_online():
                total_honest_online_devices += 1
                accuracy, losses = device.validate_model_weights()

                total_accuracy += accuracy

        average_accuracy = total_accuracy / total_honest_online_devices

        average_accuracies.append(average_accuracy)

        for i, device in enumerate(devices_list):
            if device.is_online():
                accuracy_this_round, losses = device.validate_model_weights()
            else:
                accuracy_this_round = 0

            if device.is_malicious and device.is_online():
                device.set_contribution_value(-1)

            elif device.is_online() and not device.is_malicious:
                if device.role == "worker":
                    device.set_contribution_value(1)
                elif device.role == "validator":
                    device.set_contribution_value(2)
                elif device.role == "miner":
                    device.set_contribution_value(3)
            elif not device.is_online():
                device.set_contribution_value(0)

            with open(f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a") as file:
                is_malicious_node = "M" if device.return_is_malicious() else "B"
                file.write(f"{device.return_idx()} {device.return_role()} {is_malicious_node}:{accuracy_this_round}\n")
                if i == (len(devices_list) - 1):
                    file.write(f"average_Accuracy: {average_accuracy}\n")

        print(''' Logging Stake by Devices ''')
        for device in devices_list:
            accuracy_this_round = device.validate_model_weights()
            with open(f"{log_files_folder_path_comm_round}/stake_comm_{comm_round}.txt", "a") as file:
                is_malicious_node = "M" if device.return_is_malicious() else "B"
                file.write(
                    f"{device.return_idx()} {device.return_role()} {is_malicious_node}: {device.return_stake()}\n")

        file_blockchain_size = open(f"{bench_folder}/blockchain.txt", "a")
        for device in devices_list:
            blockchain_object = device.return_blockchain_object()

            file_blockchain_size.write(str(sys.getsizeof(blockchain_object)) + "\n")

        if args['save_network_snapshots'] and (comm_round == 1 or comm_round % args['save_freq'] == 0):
            if args['save_most_recent']:
                paths = sorted(Path(network_snapshot_save_path).iterdir(), key=os.path.getmtime)
                if len(paths) > args['save_most_recent']:
                    for _ in range(len(paths) - args['save_most_recent']):
                        open(paths[_], 'w').close()
                        os.remove(paths[_])
            snapshot_file_path = f"{network_snapshot_save_path}/snapshot_r_{comm_round}"
            print(f"Saving network snapshot to {snapshot_file_path}")
            pickle.dump(devices_in_network, open(snapshot_file_path, "wb"))
