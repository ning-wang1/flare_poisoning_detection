#########################
# Purpose: Main function to perform federated training and all model poisoning attacks
########################



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
from multiprocessing import Process, Manager

import tensorflow.compat.v1 as tf

from utils.io_utils import data_setup, mal_data_setup
import global_vars as gv
from agents import agent, master
from utils.eval_utils import eval_func
from detect import Detect, save_statistics
from agg_alg import avg_agg, krum_agg, coomed_agg, trimmed_mean, bulyan_agg, soft_agg
from malicious_agent import mal_agent, mal_agent_copy
from attack import load_ben_delta, attack_krum, attack_trimmedmean


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.disable_eager_execution()


def write_dict(return_dict, keys, values):
    for key in keys:
        return_dict[str(key)] = values[key]
    return


def train_fn(X_train_shards, Y_train_shards, X_test, Y_test, return_dict,
             mal_data_X=None, mal_data_Y=None, mal_data_X_test=None, mal_data_Y_test=None):
    """
    Training process at of federated learning including agent train and aggregation
    :param X_train_shards:
    :param Y_train_shards:
    :param X_test:
    :param Y_test:
    :param return_dict:
    :param mal_data_X:
    :param mal_data_Y:
    :return:
    """

    num_agents_per_time = int(args.C * args.k)
    simul_agents = gv.num_gpus * gv.max_agents_per_gpu

    simul_num = min(num_agents_per_time, simul_agents)
    agent_indices = np.arange(args.k)
    mal_agent_index = gv.mal_agent_index

    t = 0
    mal_visible = []
    eval_loss_list = []
    lr = args.eta

    global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)

    # Start the training process
    while t < args.T:
        print('Time step %s' % t)
        process_list = []
        mal_active = 0
        curr_agents = np.random.choice(agent_indices, num_agents_per_time, replace=False)
        print('Set of agents chosen: %s' % curr_agents)
        k = 0
        agents_left = 1e4

        # untargeted attack: purely preventing from good model
        if 'untargeted' in args.attack_type:
            while k < num_agents_per_time:
                true_simul = min(simul_num, agents_left)
                print('training %s agents' % true_simul)
                for l in range(true_simul):
                    gpu_index = int(l / gv.max_agents_per_gpu)
                    gpu_id = gv.gpu_ids[gpu_index]

                    i = curr_agents[k]
                    p = Process(target=agent, args=(i, X_train_shards[i], Y_train_shards[i], t, gpu_id,
                                                    return_dict, X_test, Y_test, lr))
                    mal_active = 1
                    p.start()
                    process_list.append(p)
                    k += 1
                for item in process_list:
                    item.join()
                agents_left = num_agents_per_time - k
                print('Agents left:%s' % agents_left)

            # time.sleep(2)
            local_grads = load_ben_delta(curr_agents, t)

            if 'krum' in args.attack_type:
                p = Process(target=attack_krum, args=(args, global_weights, local_grads,
                                                          mal_agent_index, return_dict))
            elif 'trimmedmean' in args.attack_type:
                p = Process(target=attack_trimmedmean, args=(global_weights, local_grads,
                                                             mal_agent_index, return_dict))

            p.start()
            process_list.append(p)

            del local_grads

        # targeted attacks like backdoor or no attack
        else:
            while k < num_agents_per_time:
                true_simul = min(simul_num, agents_left)
                print('training %s agents' % true_simul)

                for l in range(true_simul):
                    p_start = False
                    gpu_index = int(l / gv.max_agents_per_gpu)
                    gpu_id = gv.gpu_ids[gpu_index]
                    i = curr_agents[k]

                    if args.mal is False or i not in mal_agent_index:
                        p = Process(target=agent, args=(i, X_train_shards[i], Y_train_shards[i], t, gpu_id,
                                                        return_dict, X_test, Y_test, lr))
                        p_start = True
                    elif args.mal is True and i in mal_agent_index:
                        if i == mal_agent_index[0]:
                            p = Process(target=mal_agent, args=(i, X_train_shards[i], Y_train_shards[i], mal_data_X,
                                                                mal_data_Y, t, gpu_id, return_dict, mal_visible, X_test,
                                                                Y_test))
                            p_start = True
                        mal_active = 1

                    if p_start:
                        p.start()
                        process_list.append(p)
                    k += 1
                for item in process_list:
                    item.join()
                agents_left = num_agents_per_time - k
                print('Agents left:%s' % agents_left)

            if args.mal is True:
                for i in mal_agent_index[1:]:
                    p_copy = Process(target=mal_agent_copy, args=(i, X_train_shards[i], Y_train_shards[i],
                                                                  mal_data_X, mal_data_Y, t, gpu_id, return_dict,
                                                                  mal_visible, X_test, Y_test))
                    p_copy.start()
                    p_copy.join()

        print('Joined all processes for time step %s' % t)
        if mal_active == 1:
            mal_visible.append(t)

        # detecting malicious updates
        detect = Detect(args, X_test, Y_test, curr_agents, global_weights)
        if args.detect:
            print('Detect method: {}'.format(args.detect_method))

            p_check = Process(target=detect.penul_check, args=(t, return_dict))

            p_check.start()
            p_check.join()
            alpha = return_dict['detected_mal_agent']
            global_weights = soft_agg(curr_agents, return_dict, global_weights, alpha)

        # other aggregation rules
        else:
            print('Aggregation Rule: {}'.format(args.gar))
            if 'avg' in args.gar:
                global_weights = avg_agg(args, curr_agents, return_dict, global_weights, t)
            elif 'krum' in args.gar:
                global_weights, mal_selected = krum_agg(args, return_dict, global_weights)
            elif 'coomed' in args.gar:
                global_weights = coomed_agg(curr_agents, return_dict, global_weights)
            elif 'trimmedmean' in args.gar:
                global_weights = trimmed_mean(curr_agents, return_dict, global_weights)
            elif 'bulyan' in args.gar:
                global_weights = bulyan_agg(curr_agents, return_dict, global_weights)

        # Saving global weights for the next update
        print('save current global model')
        np.save(gv.dir_name + 'global_weights_t%s.npy' % (t + 1), global_weights)

        # Evaluate global weight
        if args.mal and 'untargeted' in args.attack_type:
            p_eval = Process(target=eval_func, args=(X_test, Y_test, t + 1, return_dict),
                             kwargs={'global_weights': global_weights})
        else:
            p_eval = Process(target=eval_func,
                             args=(X_test, Y_test, t + 1, return_dict, mal_data_X, mal_data_Y),
                             kwargs={'global_weights': global_weights})
        p_eval.start()
        p_eval.join()
        eval_loss_list.append(return_dict['eval_loss'])
        print('test complete')

        # save statistics
        if args.mal and 'backdoor' in args.attack_type:
            if 'single' in args.mal_obj:
                statistics = [return_dict['agents'], gv.mal_agent_index, return_dict['agent_acc'],
                              return_dict['eval_success'], return_dict['target_conf'], return_dict['attack_succeed']]
            elif 'multiple' in args.mal_obj or 'target_backdoor':
                statistics = [return_dict['agents'], gv.mal_agent_index, return_dict['agent_acc'],
                              return_dict['eval_success'], return_dict['mal_suc_count']]
            save_statistics(gv.acc_file_dir, statistics, t == 0)
            return_dict['agent_acc'] = []
            return_dict['agents'] = []
        # elif 'untargeted' in args.attack_type:
        else:
            statistics = [return_dict['eval_success'], return_dict['eval_loss']]
            save_statistics(gv.acc_file_dir, statistics, t==0)
        t += 1
    return t


def main(args):
    """
    The main function. You can either train or test the FL
    :return:
    """
    X_train, Y_train, Y_train_uncat, X_test, Y_test, Y_test_uncat = data_setup()

    # Create data shards
    random_indices = np.random.choice(len(X_train), len(X_train), replace=False)
    X_train_permuted = X_train[random_indices]
    Y_train_permuted = Y_train[random_indices]
    X_train_shards = np.split(X_train_permuted, args.k)
    Y_train_shards = np.split(Y_train_permuted, args.k)

    # Load malicious data
    if args.mal and 'backdoor' in args.attack_type:
        mal_data_X_train, mal_data_Y_train, true_labels_train = mal_data_setup(X_train, Y_train, Y_train_uncat,
                                                                               mal_num=20, gen_flag=True)
        if 'target_backdoor' in args.mal_obj:
            mal_data_X_test, _, true_labels_test = mal_data_setup(X_test, Y_test, Y_test_uncat, mal_num=100)
            mal_data_Y_test = np.ones(len(true_labels_test)) * mal_data_Y_train[0]
        else:
            mal_data_X_test = mal_data_X_train
            mal_data_Y_test = mal_data_Y_train

    # train
    if args.train:
        p0 = Process(target=master)
        p0.start()
        p0.join()

        manager = Manager()
        return_dict = manager.dict()
        return_dict['eval_success'] = 0.0
        return_dict['eval_loss'] = 0.0
        return_dict['agent_acc'] = []
        return_dict['agent_loss'] = []
        return_dict['agents'] = []

        if args.mal and 'backdoor' in args.attack_type:
            return_dict['mal_suc_count'] = 0
            t_final = train_fn(X_train_shards, Y_train_shards, X_test, Y_test_uncat,
                               return_dict, mal_data_X_train, mal_data_Y_train,  mal_data_X_test, mal_data_Y_test)
            print('Malicious agent succeeded in %s of %s iterations' %
                  (return_dict['mal_suc_count'], t_final * args.mal_num))
        else:
            _ = train_fn(X_train_shards, Y_train_shards, X_test, Y_test_uncat, return_dict)

    # test
    else:
        manager = Manager()
        return_dict = manager.dict()
        return_dict['eval_success'] = 0.0
        return_dict['eval_loss'] = 0.0
        if args.mal:
            return_dict['mal_suc_count'] = 0
        for t in range(args.T):
            if not os.path.exists(gv.dir_name + 'global_weights_t%s.npy' % t):
                print('No directory found for iteration %s' % t)
                break
            if args.mal:
                p_eval = Process(target=eval_func, args=(X_test, Y_test_uncat, t, return_dict,
                                                         mal_data_X_test, mal_data_Y_test))
            else:
                p_eval = Process(target=eval_func, args=(X_test, Y_test_uncat, t, return_dict))

            p_eval.start()
            p_eval.join()

        if args.mal and 'backdoor' in args.attack_type:
            print('Malicious agent succeeded in %s of %s iterations' %
                  (return_dict['mal_suc_count'], (t-1) * args.mal_num))
            print('Detection succeeded in %s of %s iterations'%(return_dict['detect_suc_count'], (t-1) * args.mal_num))


if __name__ == "__main__":
    gv.init()
    seed = 3
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = gv.args
    main(args)
