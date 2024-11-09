import numpy as np
import os
import argparse
import tensorflow as tf
import global_vars as gv
from utils.io_utils import file_write
import os
import warnings
import copy
import random
from scipy.spatial import distance
from agg_alg import krum_agg, krum_one_layer

from utils.eval_utils import mal_eval_multiple, mal_eval_single, eval_minimal

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_ben_delta(current_agents, t):
    """
    estimate the delta of other agents. accuracy: L2 norm of current round estimation minus previous estimation
    :param mal_visible:
    :param t:
    :return:
    """
    args = gv.args
    delta = np.load(gv.dir_name + 'ben_delta_%s_t%s.npy' % (0, t), allow_pickle=True)

    # prepare data structures to store local gradients
    local_grads = []
    for i in range(len(current_agents)):
        local_grads.append([])
        for p in list(delta):
            local_grads[i].append(np.zeros(p.data.shape))

    for c in current_agents:
        delta_other = np.load(gv.dir_name + 'ben_delta_%s_t%s.npy' % (c, t), allow_pickle=True)
        for idx, p in enumerate(delta_other):
            local_grads[c][idx] = p

    return local_grads


# Trimmed Mean Attack--Main function for TMA.
def attack_trimmedmean(shared_weights, local_grads, mal_index, return_dict, b=2):
    benign_max = []
    benign_min = []
    average_sign = []
    mal_param = []
    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]
    for p in list(shared_weights):
        benign_max.append(np.zeros(p.shape))
        benign_min.append(np.zeros(p.shape))
        average_sign.append(np.zeros(p.shape))
        mal_param.append(np.zeros(p.shape))

    # use local grad to infer sign
    for idx, p in enumerate(average_sign):
        for c in range(len(local_param)):
            average_sign[idx] += local_param[c][idx]
        average_sign[idx] = np.sign(average_sign[idx])

    # get the min and max value of local model
    for idx, p in enumerate(shared_weights):
        temp = []
        for c in range(len(local_param)):
            local_param[c][idx] = p - local_param[c][idx]
            temp.append(local_param[c][idx])
        temp = np.array(temp)
        benign_max[idx] = np.amax(temp, axis=0)
        benign_min[idx] = np.amin(temp, axis=0)

    for idx, p in enumerate(average_sign):
        for aver_sign, b_max, b_min, mal_p in np.nditer([p, benign_max[idx], benign_min[idx], mal_param[idx]],
                                                        op_flags=['readwrite']):
            if aver_sign < 0:
                if b_min > 0:
                    mal_p[...] = random.uniform(b_min / b, b_min)
                else:
                    mal_p[...] = random.uniform(b_min * b, b_min)
            else:
                if b_max > 0:
                    mal_p[...] = random.uniform(b_max, b_max * b)
                else:
                    mal_p[...] = random.uniform(b_max, b_max / b)
    for c in mal_index:
        for idx, p in enumerate(shared_weights):
            local_grads[c][idx] = -mal_param[idx] + p
        return_dict[str(c)] = np.array(local_grads[c])

    return


def attack_krum(args, shared_weights, local_grads, mal_index, return_dict):
    for idx, _ in enumerate(local_grads[0]):
        local_grads = attack_krum_idx(args, shared_weights, local_grads, mal_index, idx)
        # return local_grads
    for c in mal_index:
        return_dict[str(c)] = np.array(local_grads[c])
    return


# Krum Attack--Main function for KA.
def attack_krum_idx(args, shared_weights, local_grads, mal_index, param_index,
                    lower_bound=1e-8, upper_bound=1e-3):
    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]
    m = len(local_grads)
    c = len(mal_index)
    d = local_grads[0][param_index].size

    average_sign = np.zeros(list(shared_weights)[param_index].shape)
    benign_max = np.zeros(list(shared_weights)[param_index].shape)

    for c in range(len(local_param)):
        average_sign += local_param[c][param_index]
    average_sign = np.sign(average_sign)
    min_dis = np.inf
    max_dis = -np.inf
    for i in range(m):
        if i in mal_index:
            continue
        else:
            temp_min_dis = 0
            temp_max_dis = 0
            for j in range(m):
                if j in mal_index or j == i:
                    continue
                else:
                    temp_min_dis += distance.euclidean(local_grads[i][param_index].flatten(),
                                                       local_grads[j][param_index].flatten())
        temp_max_dis += distance.euclidean(local_grads[i][param_index].flatten(), benign_max.flatten())

        if temp_min_dis < min_dis:
            min_dis = temp_min_dis
        if temp_max_dis > max_dis:
            max_dis = temp_max_dis

    # upper_bound = 1.0 / (m - 2*c - 1) / np.sqrt(d) * min_dis + 1.0 / np.sqrt(d) * max_dis
    upper_bound = 1.0
    lambda1 = upper_bound

    if upper_bound < lower_bound:
        print('Wrong lower bound!')

    while True:
        krum_local = []
        for kk in range(len(local_grads)):
            krum_local.append(local_grads[kk][param_index])
        for kk in mal_index:
            krum_local[kk] = -lambda1 * average_sign

        choose_index, distance_score = krum_one_layer(krum_local)
        if choose_index in mal_index:
            print('====================================> Found a lambda')
            # print('Distance scores of agents: {}, \nThe selected agent: {}'.format(distance_score, choose_index))
            break
        elif lambda1 < lower_bound:
            print(choose_index, '=======================================> Failed to find a proper lambda!')
            break
        else:
            lambda1 /= 2.0

    for kk in mal_index:
        local_grads[kk][param_index] = -lambda1 * average_sign

    return local_grads


def list_to_dict(ls):
    dic = dict()
    for idx, val in enumerate(ls):
        dic[str(idx)] = val

    return dic


def eval_mal(args, t, final_weights, X_test, Y_test,  mal_data_X=None, mal_data_Y=None):
    # evaluate malicious agent
    print('-------Eval at mal agent-------')
    if 'backdoor' in args.attack_type:
        if 'single' in args.mal_obj:
            target, target_conf, actual, actual_conf = mal_eval_single(mal_data_X, mal_data_Y, final_weights)
            print('For iter %s, Target:%s with conf. %s, Curr_pred on malicious model:%s with conf. %s' % (
                t, target, target_conf, actual, actual_conf))
        elif 'multiple' in args.mal_obj:
            suc_count_local = mal_eval_multiple(mal_data_X, mal_data_Y, final_weights)
            print('%s of %s targets achieved' % (suc_count_local, args.mal_num))

    eval_success, eval_loss = eval_minimal(X_test, Y_test, final_weights)
    print('Malicious Agent: success {}, loss {}'.format(eval_success, eval_loss))
    return eval_success
