import numpy as np
import tensorflow.compat.v1 as tf

# import tensorflow as tf
from multiprocessing import Process, Manager
from utils.io_utils import data_setup, mal_data_setup
import global_vars as gv
from agents import agent, master
from utils.eval_utils import eval_func
from malicious_agent import mal_agent
from utils.dist_utils import collate_weights, model_shape_size
from detect import flatten_nn_param
import math
import os
import warnings


def avg_agg(args, curr_agents, return_dict, global_weights, t, detected_agent=None):

    num_agents_per_time = len(curr_agents)
    if detected_agent is not None:
        alpha_i = 1.0/(args.k - len(detected_agent))
    else:
        alpha_i = 1.0 / args.k

    if args.mal:
        mal_agent_index = gv.mal_agent_index
    count = 0

    # average Aggregation
    if detected_agent is None:
        if args.mal:
            for k in range(num_agents_per_time):
                if curr_agents[k] not in mal_agent_index:
                    if count == 0:
                        ben_delta = alpha_i * return_dict[str(curr_agents[k])]
                        # np.save(gv.dir_name + 'ben_delta_sample%s.npy' % t, return_dict[str(curr_agents[k])])
                        count += 1
                    else:
                        ben_delta += alpha_i * return_dict[str(curr_agents[k])]

            np.save(gv.dir_name + 'ben_delta_t%s.npy' % t, ben_delta)
            for id in mal_agent_index:
                global_weights += alpha_i * return_dict[str(id)]
            global_weights += ben_delta
        else:
            for k in range(num_agents_per_time):
                global_weights += alpha_i * return_dict[str(curr_agents[k])]
    else:
        for k in range(num_agents_per_time):
            if curr_agents[k] not in detected_agent:
                if count == 0:
                    ben_delta = alpha_i * return_dict[str(curr_agents[k])]
                    count += 1
                else:
                    ben_delta += alpha_i * return_dict[str(curr_agents[k])]

        np.save(gv.dir_name + 'ben_delta_t%s.npy' % t, ben_delta)
        global_weights += ben_delta
    return global_weights


def krum_agg(args, return_dict, global_weights, update_global=True):
    num_agents_per_time = int(args.k * args.C)
    collated_weights = []
    collated_bias = []
    agg_num = int(num_agents_per_time - 1 - 2)

    for k in range(num_agents_per_time):
        weights_curr, bias_curr = collate_weights(return_dict[str(k)])
        collated_weights.append(weights_curr)
        collated_bias.append(collated_bias)
    score_array = np.zeros(num_agents_per_time)

    for k in range(num_agents_per_time):
        dists = []
        for i in range(num_agents_per_time):
            if i == k:
                continue
            else:
                dists.append(np.linalg.norm(collated_weights[k] - collated_weights[i]))
        dists = np.sort(np.array(dists))
        dists_subset = dists[:agg_num]
        score_array[k] = np.sum(dists_subset)

    krum_index = np.argmin(score_array)

    if update_global:
        global_weights += return_dict[str(krum_index)]
        # print('krum_index:  {}, mal_agent_index: {}'.format(krum_index, mal_agent_index))
        print('Distance scores of agents: {}, \nThe selected agent: {}'.format(score_array, krum_index))
    return global_weights, krum_index


def krum_one_layer(grad_list):
    num_agents_per_time = len(grad_list)
    agg_num = int(num_agents_per_time - 1 - 2)
    score_array = np.zeros(num_agents_per_time)

    for k in range(num_agents_per_time):
        dists = []
        for i in range(num_agents_per_time):
            if i == k:
                continue
            else:
                dists.append(np.linalg.norm(grad_list[k] - grad_list[i]))
        dists = np.sort(np.array(dists))
        dists_subset = dists[:agg_num]
        score_array[k] = np.sum(dists_subset)

    krum_index = np.argmin(score_array)

    return krum_index, score_array


def coomed_agg(curr_agents, return_dict, global_weights):
    '''
    coordinate-wise median
    :param args:
    :param curr_agents:
    :param return_dict:
    :param global_weights:
    :return:
    '''
    num_agents_per_time = len(curr_agents)

    # Fix for mean aggregation first!
    weight_tuple_0 = return_dict[str(curr_agents[0])]
    weights_0, bias_0 = collate_weights(weight_tuple_0)
    weights_array = np.zeros((num_agents_per_time, len(weights_0)))
    bias_array = np.zeros((num_agents_per_time, len(bias_0)))
    shape_size = model_shape_size(weight_tuple_0)

    for k in range(num_agents_per_time):
        weight_tuple = return_dict[str(curr_agents[k])]
        weights_curr, bias_curr = collate_weights(weight_tuple)
        weights_array[k, :] = weights_curr
        bias_array[k, :] = bias_curr

    med_weights = np.median(weights_array, axis=0)
    med_bias = np.median(bias_array, axis=0)
    num_layers = len(shape_size[0])
    update_list = []
    w_count = 0
    b_count = 0

    for i in range(num_layers):
        weights_length = shape_size[2][i]
        update_list.append(med_weights[w_count: w_count + weights_length].reshape(shape_size[0][i]))
        w_count += weights_length
        bias_length = shape_size[3][i]
        update_list.append(med_bias[b_count:b_count + bias_length].reshape(shape_size[1][i]))
        b_count += bias_length
    assert model_shape_size(update_list) == shape_size
    global_weights += update_list

    return global_weights


def t_mean(samples, beta):
    size = samples.shape[0]
    beyond_choose = int(size * beta)
    samples = np.sort(samples, axis=0)
    samples = samples[beyond_choose:size - beyond_choose]
    average_grad = np.average(samples, axis=0)

    return average_grad


def trimmed_mean(curr_agents, return_dict, global_weights, beta=0.1):

    num_agents_per_time = len(curr_agents)

    # Fix for mean aggregation first!
    weight_tuple_0 = return_dict[str(curr_agents[0])]
    weights_0, bias_0 = collate_weights(weight_tuple_0)
    weights_array = np.zeros((num_agents_per_time, len(weights_0)))
    bias_array = np.zeros((num_agents_per_time, len(bias_0)))
    shape_size = model_shape_size(weight_tuple_0)

    for k in range(num_agents_per_time):
        weight_tuple = return_dict[str(curr_agents[k])]
        weights_curr, bias_curr = collate_weights(weight_tuple)
        weights_array[k, :] = weights_curr
        bias_array[k, :] = bias_curr

    mean_weights = t_mean(weights_array, beta)
    mean_bias = t_mean(bias_array, beta)
    num_layers = len(shape_size[0])
    update_list = []
    w_count = 0
    b_count = 0

    for i in range(num_layers):
        weights_length = shape_size[2][i]
        update_list.append(mean_weights[w_count: w_count + weights_length].reshape(shape_size[0][i]))
        w_count += weights_length
        bias_length = shape_size[3][i]
        update_list.append(mean_bias[b_count:b_count + bias_length].reshape(shape_size[1][i]))
        b_count += bias_length
    assert model_shape_size(update_list) == shape_size
    global_weights += update_list

    return global_weights


def bulyan_agg(curr_agents, return_dict, global_weights):
    grad_list = [return_dict[str(i)] for i in curr_agents]
    average_grad = []
    for p in list(grad_list[0]):
        # print(p.data.shape)
        average_grad.append(np.zeros(p.data.shape))

    for idx, _ in enumerate(grad_list[0]):
        bulyan_local = []
        for kk in range(len(curr_agents)):
            bulyan_local.append(grad_list[kk][idx])
        average_grad[idx] = bulyan(bulyan_local, aggsubfunc='krum')

    global_weights += average_grad
    return global_weights


def bulyan(grads, aggsubfunc='trimmedmean', f=1):
    samples = np.array(grads)
    feature_shape = grads[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())

    grads_num = len(samples_flatten)
    theta = grads_num - 2 * f
    # bulyan cannot do the work here when theta <= 0. Actually, it assumes n >= 4 * f + 3
    selected_grads = []
    # here, we use krum as sub algorithm
    if aggsubfunc == 'krum':
        for i in range(theta):
            krum_grad, _ = krum_1(samples_flatten)
            selected_grads.append(krum_grad)
            for j in range(len(samples_flatten)):
                if samples_flatten[j] is krum_grad:
                    del samples_flatten[j]
                    break
    elif aggsubfunc == 'trimmedmean':
        for i in range(theta):
            trimmedmean_grads = trimmed_mean_1(samples_flatten)
            selected_grads.append(trimmedmean_grads)
            min_dis = np.inf
            min_index = None
            for j in range(len(samples_flatten)):
                temp_dis = np.linalg.norm(trimmedmean_grads - samples_flatten[j])
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    min_index = j
            assert min_index != None
            del samples_flatten[min_index]

    beta = theta - 2 * f
    np_grads = np.array([g.flatten().tolist() for g in selected_grads])

    grads_dim = len(np_grads[0])
    selected_grads_by_cod = np.zeros([grads_dim, 1])  # shape of torch grads
    for i in range(grads_dim):
        selected_grads_by_cod[i, 0] = bulyan_one_coordinate(np_grads[:, i], beta)

    return selected_grads_by_cod.reshape(feature_shape)


def trimmed_mean_1(samples, beta=0.1):
    samples = np.array(samples)
    average_grad = np.zeros(samples[0].shape)
    size = samples.shape[0]
    beyond_choose = int(size * beta)
    samples = np.sort(samples, axis=0)
    samples = samples[beyond_choose:size-beyond_choose]
    average_grad = np.average(samples, axis=0)

    return average_grad


def krum_1(samples, f=0):
    size = len(samples)
    size_ = size - f - 2
    metric = []
    for idx in range(size):
        sample = samples[idx]
        samples_ = samples.copy()
        del samples_[idx]
        dis = np.array([np.linalg.norm(sample-sample_) for sample_ in samples_])
        metric.append(np.sum(dis[np.argsort(dis)[:size_]]))
    index = np.argmin(metric)
    return samples[index], index


def bulyan_median(arr):
    arr_len = len(arr)
    distances = np.zeros([arr_len, arr_len])
    for i in range(arr_len):
        for j in range(arr_len):
            if i < j:
                distances[i, j] = abs(arr[i] - arr[j])
            elif i > j:
                distances[i, j] = distances[j, i]
    total_dis = np.sum(distances, axis=-1)
    median_index = np.argmin(total_dis)
    return median_index, distances[median_index]


def bulyan_one_coordinate(arr, beta):
    _, distances = bulyan_median(arr)
    median_beta_neighbors = arr[np.argsort(distances)[:beta]]
    return np.mean(median_beta_neighbors)


def soft_agg(curr_agents, return_dict, global_weights, alpha):
    """

    :param curr_agents:
    :param return_dict:
    :param global_weights:
    :param alpha: trust scores
    :return:
    """

    num_agents_per_time = len(curr_agents)

    # soft Aggregation
    for k in range(num_agents_per_time):
        global_weights += alpha[curr_agents[k]] * return_dict[str(curr_agents[k])]

    return global_weights


def soft_agg_norm(curr_agents, return_dict, global_weights, alpha):
    """

    :param curr_agents:
    :param return_dict:
    :param global_weights:
    :param t:
    :param alpha: trust scores
    :return:
    """

    num_agents_per_time = len(curr_agents)

    clean_update = return_dict[str(num_agents_per_time)]
    clean_update_flatten = flatten_nn_param(clean_update)
    norm_clean = np.linalg.norm(clean_update_flatten)

    # soft Aggregation
    for k in range(num_agents_per_time):
        update = return_dict[str(curr_agents[k])]
        update_flatten = flatten_nn_param(update)
        norm_update = np.linalg.norm(update_flatten)
        global_weights += alpha[curr_agents[k]] * update * norm_clean / norm_update

    return global_weights



# def bulyan_agg(curr_agents, return_dict, global_weights):
#     num_agents_per_time = len(curr_agents)
#
#     # Fix for mean aggregation first!
#     weight_tuple_0 = return_dict[str(curr_agents[0])]
#     weights_0, bias_0 = collate_weights(weight_tuple_0)
#     weights_array = np.zeros((num_agents_per_time, len(weights_0)))
#     bias_array = np.zeros((num_agents_per_time, len(bias_0)))
#     shape_size = model_shape_size(weight_tuple_0)
#
#     for k in range(num_agents_per_time):
#         weight_tuple = return_dict[str(curr_agents[k])]
#         weights_curr, bias_curr = collate_weights(weight_tuple)
#         weights_array[k, :] = weights_curr
#         bias_array[k, :] = bias_curr
#
#     bulyan_weights = bulyan(weights_array)
#     bulyan_bias = bulyan(bias_array)
#     num_layers = len(shape_size[0])
#     update_list = []
#     w_count = 0
#     b_count = 0
#
#     for i in range(num_layers):
#         weights_length = shape_size[2][i]
#         update_list.append(bulyan_weights[w_count: w_count + weights_length].reshape(shape_size[0][i]))
#         w_count += weights_length
#         bias_length = shape_size[3][i]
#         update_list.append(bulyan_bias[b_count:b_count + bias_length].reshape(shape_size[1][i]))
#         b_count += bias_length
#     assert model_shape_size(update_list) == shape_size
#     global_weights += update_list
#
#     return global_weights
#
#
# def bulyan(samples, f=1):
#
#     grads_num = len(samples)
#     theta = grads_num - 2 * f
#     # bulyan cannot do the work here when theta <= 0. Actually, it assumes n >= 4 * f + 3
#     selected_grads = []
#     # here, we use krum as sub algorithm
#
#     for i in range(theta):
#         krum_idx, _ = krum_one_layer(samples)
#         selected_grads.append(samples[krum_idx])
#         for j in range(len(samples)):
#             if samples[j] is krum_idx:
#                 del samples[j]
#                 break
#
#     beta = theta - 2 * f
#     np_grads = np.array(selected_grads)
#     grad = t_mean(np_grads, beta)
#
#     return grad
#

# def bulyan(grads, f=1):
#     def bulyan_median(arr):
#         arr_len = len(arr)
#         distances = np.zeros([arr_len, arr_len])
#         for i in range(arr_len):
#             for j in range(arr_len):
#                 if i < j:
#                     distances[i, j] = abs(arr[i] - arr[j])
#                 elif i > j:
#                     distances[i, j] = distances[j, i]
#         total_dis = np.sum(distances, axis=-1)
#         median_index = np.argmin(total_dis)
#         return median_index, distances[median_index]
#
#     samples = np.array(grads)
#     feature_shape = grads[0].shape
#     samples_flatten = []
#     for i in range(samples.shape[0]):
#         samples_flatten.append(samples[i].flatten())
#
#     grads_num = len(samples_flatten)
#     theta = grads_num - 2 * f
#     # bulyan cannot do the work here when theta <= 0. Actually, it assumes n >= 4 * f + 3
#     selected_grads = []
#     # here, we use krum as sub algorithm
#
#     for i in range(theta):
#         krum_grad, _ = krum_one_layer(samples_flatten)
#         selected_grads.append(krum_grad)
#         for j in range(len(samples_flatten)):
#             if samples_flatten[j] is krum_grad:
#                 del samples_flatten[j]
#                 break
#
#     beta = theta - 2 * f
#     np_grads = np.array([g.flatten().tolist() for g in selected_grads])
#
#     grads_dim = len(np_grads[0])
#     selected_grads_by_cod = np.zeros([grads_dim, 1])  # shape of grads
#     for i in range(grads_dim):
#         layer_grad = np_grads[:, i]
#         _, distances = bulyan_median(layer_grad)
#         median_beta_neighbors = layer_grad[np.argsort(distances)[:beta]]
#         selected_grads_by_cod[i, 0] = np.mean(median_beta_neighbors)
#     # grad = selected_grads_by_cod.reshape(feature_shape)
#
#     return selected_grads_by_cod














 # if 'avg' in args.gar:
 #            if args.mal:
 #                count = 0
 #                for k in range(num_agents_per_time):
 #                    if curr_agents[k] != mal_agent_index:
 #                        if count == 0:
 #                            ben_delta = alpha_i * return_dict[str(curr_agents[k])]
 #                            np.save(gv.dir_name + 'ben_delta_sample%s.npy' % t, return_dict[str(curr_agents[k])])
 #                            count += 1
 #                        else:
 #                            ben_delta += alpha_i * return_dict[str(curr_agents[k])]
 #
 #                np.save(gv.dir_name + 'ben_delta_t%s.npy' % t, ben_delta)
 #                global_weights += alpha_i * return_dict[str(mal_agent_index)]
 #                global_weights += ben_delta
 #            else:
 #                for k in range(num_agents_per_time):
 #                    global_weights += alpha_i * return_dict[str(curr_agents[k])]
 #
 #        elif 'krum' in args.gar:
 #            collated_weights = []
 #            collated_bias = []
 #            agg_num = int(num_agents_per_time-1-2)
 #
 #            for k in range(num_agents_per_time):
 #                weights_curr, bias_curr = collate_weights(return_dict[str(k)])
 #                collated_weights.append(weights_curr)
 #                collated_bias.append(collated_bias)
 #            score_array = np.zeros(num_agents_per_time)
 #
 #            for k in range(num_agents_per_time):
 #                dists = []
 #                for i in range(num_agents_per_time):
 #                    if i == k:
 #                        continue
 #                    else:
 #                        dists.append(np.linalg.norm(collated_weights[k]-collated_weights[i]))
 #                dists = np.sort(np.array(dists))
 #                dists_subset = dists[:agg_num]
 #                score_array[k] = np.sum(dists_subset)
 #
 #            krum_index = np.argmin(score_array)
 #            print('distance scores of agents: {}, \n the selected agent: {}'.format(score_array, krum_index))
 #
 #            global_weights += return_dict[str(krum_index-1)]
 #            if krum_index == mal_agent_index:
 #                krum_select_indices.append(t)
 #
 #        elif 'coomed' in args.gar:
 #            # Fix for mean aggregation first!
 #            weight_tuple_0 = return_dict[str(curr_agents[0])]
 #            weights_0, bias_0 = collate_weights(weight_tuple_0)
 #            weights_array = np.zeros((num_agents_per_time, len(weights_0)))
 #            bias_array = np.zeros((num_agents_per_time, len(bias_0)))
 #            shape_size = model_shape_size(weight_tuple_0)
 #
 #            for k in range(num_agents_per_time):
 #                weight_tuple = return_dict[str(curr_agents[k])]
 #                weights_curr, bias_curr = collate_weights(weight_tuple)
 #                weights_array[k, :] = weights_curr
 #                bias_array[k, :] = bias_curr
 #
 #            med_weights = np.median(weights_array, axis=0)
 #            med_bias = np.median(bias_array, axis=0)
 #            num_layers = len(shape_size[0])
 #            update_list = []
 #            w_count = 0
 #            b_count = 0
 #
 #            for i in range(num_layers):
 #                weights_length = shape_size[2][i]
 #                update_list.append(med_weights[w_count: w_count+weights_length].reshape(shape_size[0][i]))
 #                w_count += weights_length
 #                bias_length = shape_size[3][i]
 #                update_list.append(med_bias[b_count:b_count+bias_length].reshape(shape_size[1][i]))
 #                b_count += bias_length
 #            assert model_shape_size(update_list) == shape_size
 #            global_weights += update_list