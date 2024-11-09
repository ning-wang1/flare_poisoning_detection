# Ning Wang 02/15/2022, for yolo demonstration
import numpy as np
import os
import math
import time
from utils.mmd import kernel_mmd


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


def penul_check(file_paths, num_agents_per_time, curr_agents):
    """
    evaluate the statistics of the penultimate layer valuer and detect anomaly
    :param file_paths: a list of file paths that store the plrs of agents
    :param num_agents_per_time:
    :param curr_agents:
    :return:
    """
    agents_penul_ls = []
    start = time.time()
    for k in range(num_agents_per_time):
        agent_k = curr_agents[k]
        print('Agents: ', agent_k)

        penul_ls_per_agent = np.load(file_paths[k])
        agents_penul_ls.append(penul_ls_per_agent)

    penul_ls_per_class = agents_penul_ls
    sta_ls = cal_nearest_neighbor(penul_ls_per_class, curr_agents)
    sta_arr = np.array(sta_ls).reshape([num_agents_per_time, -1])

    # return the count of each client. a larger count mean more trustworthy
    tau = 1
    # id_larger_count = [i for i in range(len(sta_ls)) if sta_ls[i]>0.5*len(sta_ls)]
    sta_ls = [len(sta_ls) if a >= 0.5*len(sta_ls) else a for a in sta_ls]
    exp_sum = np.sum([math.exp(a / tau) for a in sta_ls])
    alpha = [math.exp(a / tau) / exp_sum for a in sta_ls]
    print(alpha)
    save_path = '/home/ning/Extend/yolov5/model_weights/trust_scores.npy'

    with open(save_path, 'wb') as f:
        np.save(f, alpha)

    return


if __name__ == '__main__':
    path_common = '/home/ning/Extend/yolov5/plr/'
    # agents_num = 10
    agents_num = 15
    agents = [i for i in range(agents_num)]
    paths = []
    for idx in range(agents_num):
        paths.append(os.path.join(path_common, str(idx) + '_plr.npy'))

    penul_check(paths, agents_num, agents)