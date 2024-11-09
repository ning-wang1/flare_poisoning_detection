#########################
# Purpose: Useful functions for evaluating a model on test data
########################

import numpy as np
from numpy.linalg import norm
import random
import math
import time
# tf.set_random_seed(777)
# np.random.seed(777)
import csv
import copy
import global_vars as gv
import os
import warnings
from utils.mmd import kernel_mmd
from utils.eval_utils import eval_setup

from sklearn.decomposition import IncrementalPCA
from matplotlib import pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Detect:
    def __init__(self, args,  X_test, Y_test, current_agents, global_weights):
        self.args = args
        self.X_test = X_test
        self.Y_test = Y_test
        self.global_weights = global_weights
        self.curr_agents = current_agents
        self.num_agents_per_time = len(current_agents)

    def penul_check(self, t, return_dict):
        """
        evaluate the statistics of the penultimate layer valuer and detect anomaly
        :param t: time step t
        :param return_dict:
        """

        args = self.args
        X_test, Y_test = self.X_test, self.Y_test
        global_weights = self.global_weights
        # self.plot_2d_dif_agents(return_dict, t, mal_agents=[9], classes=[0, 1, 2], count=10)

        agents_penul_ls = []
        selected_class = 1
        ind = np.where(Y_test == selected_class)[0]
        X_test_one = X_test[ind[0:args.aux_data_num], :]

        for k in range(self.num_agents_per_time):
            agent_k = self.curr_agents[k]
            print('Agents: ', agent_k)

            weights = copy.deepcopy(global_weights)
            weights += return_dict[str(agent_k)]

            penul_ls_per_agent = self.eval_plr_per_class(weights, X_test_one)
            agents_penul_ls.append(penul_ls_per_agent)

        print('class: {}'.format(selected_class))
        penul_ls_per_class = agents_penul_ls
        count_ls = self.cal_nearest_neighbor(penul_ls_per_class, self.curr_agents)

        alpha = self.count_to_trustscore(count_ls)
        return_dict['detected_mal_agent'] = alpha

        statistics = [self.curr_agents, count_ls, gv.mal_agent_index, alpha]
        save_statistics(gv.mmd_file_dir, statistics, t == 0)

        return

    def count_to_trustscore(self, count_ls, tau=1, method='exponential'):
        """

        :param count_ls:
        :param tau: the exponential coefficient, valid only when method is 'exponential'
        :param method: {'exponential', 'detect'}
        :return:
        """
        if method is 'exponential':
            count_avg = sum(count_ls)/len(count_ls)
            # return the count of each client. a larger count mean more trustworthy
            exp_sum = np.sum([math.exp(a / (tau*count_avg)) for a in count_ls])
            alpha = [math.exp(a / (tau*count_avg)) / exp_sum for a in count_ls]
        else:
            agent_num = len(count_ls)
            print(agent_num)
            agents_top = np.argsort(count_ls)[int(0.5*agent_num):]
            alpha=[2/agent_num if a in agents_top else 0 for a in range(agent_num)]

        print('the trust scores for all : {}'.format(alpha))
        mal_id = list(set(gv.mal_agent_index))
        print('the true malicious agent is {}, its trust score: {}'.format(mal_id, [alpha[i] for i in mal_id]))

        return alpha

    @staticmethod
    def cal_nearest_neighbor(data_ls, agents_ls):
        """
        for each agent, calculate the number of times of being selected as other's nearest neighbor
        :param data_ls:
        :param agents_ls:
        :return:
        """
        srcs_num = len(data_ls)
        mmd_arr = np.zeros([srcs_num, srcs_num])
        sorted_ind = np.argsort(agents_ls)

        count_ls = []
        for src in range(srcs_num):
            data = data_ls[src]
            if src == 0:
                data_concate = data
            else:
                data_concate = np.concatenate((data_concate, data))

        # calculate mmd
        for src_1 in range(srcs_num):
            loc1 = sorted_ind[src_1]
            data_1 = data_ls[loc1]

            for src_2 in range(src_1 + 1, srcs_num):
                loc2 = sorted_ind[src_2]
                data_2 = data_ls[loc2]
                mmd2u = kernel_mmd(data_1, data_2)
                if mmd2u < 0:
                    mmd2u = 0
                mmd_arr[src_1, src_2] = mmd2u
                mmd_arr[src_2, src_1] = mmd2u
            # print('agent {}, mmd is {}'.format(src_1, mmd_arr[src_1]))

        # calculate the count that each client be selected as others' 50% nearest neighbors
        ind = np.argsort(mmd_arr, axis=1)
        k = int(0.5 * srcs_num)
        top_50 = ind[:, 1:k]
        for i in range(len(agents_ls)):
            top_count = np.sum(top_50 == i)

            count_ls.append(top_count)
        return count_ls

    def eval_plr_per_class(self, weights, x_test):
        """
        evaluate the PLR of a model (weights) on a test dataset. PLR is organized for seperate class
        :return: PLR list
        """
        x, y, sess, prediction, layer_output, loss = eval_setup(weights)

        layer_output_np_0 = sess.run(layer_output, feed_dict={x: x_test})
        layer_output_dim = layer_output_np_0.shape[1]
        layer_output_np = np.zeros((len(x_test), layer_output_dim))

        layer_output_np_i = sess.run(layer_output, feed_dict={x: x_test})
        layer_output_np[:, :] = layer_output_np_i

        penul_ls = layer_output_np
        sess.close()

        return penul_ls

    def cal_plr_multi_classes(self, return_dict, selected_class, count=10):
        """
        calculate the PLRs of different models using the same dataset.
        :param return_dict:
        :param num_agents_per_time:
        :param curr_agents:
        :param count:
        :return:
        """

        global_weights = self.global_weights
        X_test, Y_test = self.X_test, self.Y_test
        curr_agents = self.curr_agents
        num_agents_per_time = len(curr_agents)

        agents_penul_ls = []
        ind = []
        # select count in each class
        for c in selected_class:
            index = np.where(Y_test == c)[0]
            ind = ind + list(index[0:count])
        X_test = X_test[ind, :]

        for k in range(num_agents_per_time):
            agent_k = curr_agents[k]
            print('Agents: ', agent_k)

            weights = copy.deepcopy(global_weights)
            weights += return_dict[str(agent_k)]

            penul_ls_per_agent = self.eval_plr_per_class(weights, X_test)
            agents_penul_ls.append(penul_ls_per_agent)

        print('class: {}'.format(selected_class))
        return agents_penul_ls

    def plot_2d_dif_agents(self, return_dict, t, mal_agents, classes, count=10):
        """

        :param return_dict:
        :param t:
        :param mal_agents:
        :param classes:
        :param count:
        :return:
        """
        agent_num = 0
        mal_num = 0
        curr_agents = self.curr_agents

        templates = self.global_weights[-2]
        template_vecs = [templates[:, i] for i in classes]
        penul_value = self.cal_plr_multi_classes(return_dict, classes, 50)

        # find orthonormal basis of plane using gram-schmidt
        basis = self.gram_schmidt(template_vecs)  # shape: (3, D)
        markers = ['.', '1', '|']
        cumul_len = []
        points_per_agent = int(len(classes) * count)

        # get the idx for different categories
        for l in range(len(classes)):
            if l == 0:
                cumul_len.append(count)
            else:
                cumul_len.append(cumul_len[l - 1] + count)

        # there are several classes for each agents
        for i, a in enumerate(curr_agents):
            if a not in mal_agents:
                if agent_num == 0:
                    repre = copy.deepcopy(penul_value[i])
                else:
                    repre = np.concatenate((repre, penul_value[i]), axis=0)
                agent_num += 1

            else:
                if mal_num == 0:
                    mal_rep = copy.deepcopy(penul_value[i])
                else:
                    mal_rep = np.concatenate((mal_rep, penul_value[i]), axis=0)
                mal_num += 1

        X = np.concatenate((repre, mal_rep))  # (N * 3, D)

        proj_X = X @ basis.T  # (N * 3, 3)
        # proj_X_2d = PCA(n_components=2).fit_transform(proj_X)  # (N * 3, 2)
        transformer = IncrementalPCA(n_components=2, batch_size=20)
        proj_X_2d = transformer.fit_transform(proj_X)
        proj_ben = proj_X_2d[0:points_per_agent * agent_num, :]
        proj_mal = proj_X_2d[points_per_agent * agent_num:, :]
        proj_ben = np.reshape(proj_ben, (agent_num, points_per_agent, -1))
        proj_mal = np.reshape(proj_mal, (mal_num, points_per_agent, -1))

        file_path = os.getcwd()
        fig = plt.figure(figsize=(3.8, 3.5))
        plot_dis(proj_ben, proj_mal, [5, 15, 25], t)

        for i in range(len(classes)):
            for a in range(agent_num):
                x = proj_ben[a, i * count:(i + 1) * count, 0]
                y = proj_ben[a, i * count:(i + 1) * count, 1]
                if a == 0:
                    plt.scatter(x, y, c='b', marker=markers[i], label='Class {} (ben)'.format(i))
                else:
                    plt.scatter(x, y, c='b', marker=markers[i])

        for i in range(len(classes)):
            for m in range(mal_num):
                x_mal = proj_mal[m, i * count:(i + 1) * count, 0]
                y_mal = proj_mal[m, i * count:(i + 1) * count, 1]
                plt.scatter(x_mal, y_mal, c='r', marker=markers[i], label='Class {} (mal)'.format(i))

        plt.legend(borderaxespad=0.1, handletextpad=0.01, labelspacing=0.01, columnspacing=0.5, ncol=2)
        plt.tight_layout()
        plt.show()
        plt.close()
        fig.savefig(file_path + '/figures/penul{}.pdf'.format(t))

    @staticmethod
    def gram_schmidt(vectors):
        basis = []
        for v in vectors:
            w = v - np.sum(np.dot(v, b) * b for b in basis)
            if (w > 1e-10).any():
                basis.append(w / np.linalg.norm(w))
        return np.array(basis)


def flatten_nn_param(nn_param):
    """
    flatten the model parameters
    :param nn_param: the neural network weights
    :return: the flatterned model parameters
    """
    layers = len(nn_param)
    # ignore bias
    for i in range(layers):
        # print('layer {}, dimensions: {}'.format(i, nn_param[i].shape))
        if i % 2 == 0:
            if i == 0:
                one_d_param = nn_param[i].reshape([1, -1])
            else:
                one_d_param = np.concatenate((one_d_param, nn_param[i].reshape([1, -1])), axis=1)
    print('new dimensionality is ', one_d_param.shape)
    return one_d_param


def compare_w_server_pattern(t, return_dict, num_agents_per_time, curr_agents):
    """
    Baseline from FLTrust. Compare each client with the central server. Use cosine similarity as a metric.
    :param t: time step t
    :param return_dict: the dictionary used for multi thread
    :param num_agents_per_time: # selected clients in each iteration
    :param curr_agents:
    :return:
    """
    ts = np.zeros(num_agents_per_time)
    normed_ratio = np.zeros(num_agents_per_time)

    clean_update = return_dict[str(num_agents_per_time)]
    clean_update_flatten = flatten_nn_param(clean_update)
    clean_norm = norm(clean_update_flatten)
    start = time.time()

    for k in range(num_agents_per_time):
        agent_k = curr_agents[k]
        print('Agents: ', agent_k)
        update = return_dict[str(agent_k)]
        update_flatten = flatten_nn_param(update)
        update_norm = norm(update_flatten)
        ts[agent_k] = np.inner(clean_update_flatten, update_flatten) / (clean_norm * update_norm)
        if ts[agent_k] < 0:
            ts[agent_k] = -ts[agent_k]
        normed_ratio[agent_k] = clean_norm / update_norm

    print('trust score based on the cosine similarity of model updates:', ts / sum(ts))
    print('normed ratio:', normed_ratio)
    print('combined coefficient', ts / sum(ts) * normed_ratio)

    print('>>>>>>>>>>>>>>>time elapse: ', time.time() - start)

    return_dict['detected_mal_agent'] = ts / sum(ts) * normed_ratio
    detected_mal_agent = np.argsort(ts)[0]
    statistics = [curr_agents, ts, gv.mal_agent_index, detected_mal_agent]
    save_statistics(gv.mmd_file_dir, statistics, t == 0)
    return


def plot_dis(data_ben, data_mal, point_locs, t):
    x_ben = data_ben[:, :, 0]
    y_ben = data_ben[:, :, 1]

    x_mal = data_mal[:, :, 0]
    y_mal = data_mal[:, :, 1]

    ben_group = x_ben.shape[0]
    mal_group = x_mal.shape[0]

    markers = ['o', 's', 'P']

    plt.figure(figsize=(5, 3.5))

    for i, point_loc in enumerate(point_locs):
        samples = data_ben[:, point_loc, :]
        size = len(samples)
        size_ = size - 2
        metric = []
        for idx in range(size):
            sample = samples[idx]
            samples_ = samples.copy()
            id = [i for i in range(idx)] + [i for i in range(size) if i > idx]
            samples_ = samples_[id, :]
            dis = np.array([np.linalg.norm(sample - sample_) for sample_ in samples_])
            metric.append(np.sum(dis[np.argsort(dis)[:size_]]))
        center = np.argmin(metric)

        for j in range(ben_group):
            x12 = [x_ben[j][point_loc], x_ben[center][point_loc]]
            y12 = [y_ben[j][point_loc], y_ben[center][point_loc]]
            plt.plot(x12, y12, color='k')

        for j in range(mal_group):
            x13 = [x_mal[j][point_loc], x_ben[center][point_loc]]
            y13 = [y_mal[j][point_loc], y_ben[center][point_loc]]
            plt.plot(x13, y13, color='green')

        plt.scatter(x_ben[:, point_loc], y_ben[:, point_loc], color='b', marker=markers[i],
                    label='Point {} w/ ben model'.format(i))
        plt.scatter(x_mal[:, point_loc], y_mal[:, point_loc], color='r', marker=markers[i],
                    label='Point {} w/ mal model'.format(i))

    plt.legend(ncol=2, loc='upper right', bbox_to_anchor=(0.99, -0.1))
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()

    file_path = os.getcwd()
    plt.savefig(file_path + '/figures/dis{}.pdf'.format(t))
    plt.show()


def save_statistics(file_dir, line_to_add, create):
    if os.path.exists(file_dir) and not create:
        with open(file_dir, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)
    else:
        with open(file_dir, 'w') as f:
            writer = csv.writer(f)
            if 'mmd' in file_dir:
                header = ['agents', 'nearest neighbor counts', 'malicious_agents', 'trust score']
            elif 'acc' in file_dir and 'backdoor' in file_dir:
                header = ['agents', 'malicious_agents', 'agent_acc', 'server_acc',
                          'server_target_conf or success count', 'attack_succeed']
            else:
                header = ['server_acc', 'server_loss']
            writer.writerow(header)
            writer.writerow(line_to_add)

    return create
