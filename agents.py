#########################
# Purpose: Mimics a benign agent in the federated learning setting and sets up the master agent 
########################
import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
# tf.set_random_seed(777)
# np.random.seed(777)
import keras.backend as K
from utils.mnist import model_mnist
from utils.census_utils import census_model_1
from utils.kather_utils import model_kather
from utils.cifar_utils import  model_cifar

from utils.eval_utils import eval_minimal
import global_vars as gv
import copy

import warnings
warnings.filterwarnings('ignore')

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gv.mem_frac)


def agent(i, X_shard, Y_shard, t, gpu_id, return_dict, X_test, Y_test, lr=None):
    """
    :param i: index of agent
    :param X_shard: data shard
    :param Y_shard: labels of data shard
    :param t: training round of federated learning
    :param gpu_id: the GPU index in use
    :param return_dict:
    :param X_test: test data
    :param Y_test: test label
    :param lr: learning rate
    :return:
    """

    K.set_learning_phase(1)

    args = gv.args
    if lr is None:
        lr = args.eta

    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    shard_size = len(X_shard)

    if args.steps is not None:
        num_steps = args.steps
    else:
        num_steps = int(int(args.E) * shard_size / args.B)

    if args.dataset == 'census':
        x = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)
    else:
        x = tf.placeholder(shape=(None, gv.IMAGE_ROWS, gv.IMAGE_COLS, gv.NUM_CHANNELS), dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)

    if 'MNIST' in args.dataset:
        agent_model = model_mnist(type=args.model_num)
    elif args.dataset == 'census':
        agent_model = census_model_1()
    elif 'kather' in args.dataset:
        agent_model = model_kather()
    elif 'CIFAR' in args.dataset:
        agent_model = model_cifar()

    logits = agent_model(x)

    if args.dataset == 'census':
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    else:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    elif args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        sess = tf.Session(config=config)
    elif args.k == 1:
        sess = tf.Session()

    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    agent_model.set_weights(shared_weights)

    start_offset = 0
    if args.steps is not None:
        start_offset = (t * args.B * args.steps) % (shard_size - args.B)

    for step in range(num_steps):
        offset = (start_offset + step * args.B) % (shard_size - args.B)
        X_batch = X_shard[offset: (offset + args.B)]
        Y_batch = Y_shard[offset: (offset + args.B)]
        Y_batch_uncat = np.argmax(Y_batch, axis=1)
        _, loss_val = sess.run([optimizer, loss], feed_dict={x: X_batch, y: Y_batch_uncat})
        if step % 1000 == 0:
            print('Agent %s, Step %s, Loss %s, offset %s' % (i, step, loss_val, offset))

    local_weights = agent_model.get_weights()
    local_delta = local_weights - shared_weights

    # eval_success, eval_loss = eval_minimal(X_test, Y_test, shared_weights)
    # print('Agent {}: success {}, loss {}'.format(i, eval_success, eval_loss))
    eval_success, eval_loss = eval_minimal(X_test, Y_test, shared_weights+local_delta)
    print('Agent {}: success {}, loss {}'.format(i, eval_success, eval_loss))

    return_dict[str(i)] = np.array(local_delta)

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (i, t), local_delta)
    if i not in gv.mal_agent_index:
        agent_acc = copy.deepcopy(return_dict['agent_acc'])
        agent_acc.append(eval_success)
        agent_loss = copy.deepcopy(return_dict['agent_loss'])
        agent_loss.append(eval_loss)
        return_dict['agent_acc'] = agent_acc
        return_dict['agent_loss'] = agent_loss
        agents = copy.deepcopy(return_dict['agents'])
        agents.append(i)
        return_dict['agents'] = agents
    return


def agent_copy(i, X_shard, Y_shard, t, gpu_id, return_dict, X_test, Y_test, lr=None):
    """
    :param i: index of agent
    :param X_shard: data shard
    :param Y_shard: labels of data shard
    :param t: training round of federated learning
    :param gpu_id: the GPU index in use
    :param return_dict:
    :param X_test: test data
    :param Y_test: test label
    :param lr: learning rate
    :return:
    """

    K.set_learning_phase(1)

    args = gv.args
    if lr is None:
        lr = args.eta

    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    shard_size = len(X_shard)

    if args.dataset == 'census':
        x = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)
    else:
        x = tf.placeholder(shape=(None, gv.IMAGE_ROWS, gv.IMAGE_COLS, gv.NUM_CHANNELS), dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)

    if 'MNIST' in args.dataset:
        agent_model = model_mnist(type=args.model_num)
    elif args.dataset == 'census':
        agent_model = census_model_1()
    elif 'kather' in args.dataset:
        agent_model = model_kather()
    elif 'CIFAR' in args.dataset:
        agent_model = model_cifar()

    logits = agent_model(x)

    if args.dataset == 'census':
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    else:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    elif args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        sess = tf.Session(config=config)
    elif args.k == 1:
        sess = tf.Session()

    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    agent_model.set_weights(shared_weights)

    start_offset = 0
    if args.steps is not None:
        start_offset = (t * args.B * args.steps) % (shard_size - args.B)

    for step in range(500):
        offset = int(np.random.random() * shard_size)
        X_batch = X_shard[offset: (offset + args.B)]
        Y_batch_uncat = Y_shard[offset: (offset + args.B)]
        _, loss_val = sess.run([optimizer, loss], feed_dict={x: X_batch, y: Y_batch_uncat})
        if step % 1000 == 0:
            print('Agent %s, Step %s, Loss %s, offset %s' % (i, step, loss_val, offset))

    local_weights = agent_model.get_weights()
    local_delta = local_weights - shared_weights

    eval_success, eval_loss = eval_minimal(X_test, Y_test, shared_weights+local_delta)
    print('Agent {}: success {}, loss {}'.format(i, eval_success, eval_loss))

    return_dict[str(i)] = np.array(local_delta)

    return


def train_clean_model(t, gpu_id, return_dict, X, Y, lr=None):
    """
    :param t: training round of federated learning
    :param gpu_id: the GPU index in use
    :param return_dict:
    :param X_test: test data
    :param Y_test: test label
    :param lr: learning rate
    :return:
    """

    K.set_learning_phase(1)

    args = gv.args
    if lr is None:
        lr = args.eta

    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    data_size = len(Y)

    if args.dataset == 'census':
        x = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)
    else:
        x = tf.placeholder(shape=(None, gv.IMAGE_ROWS, gv.IMAGE_COLS, gv.NUM_CHANNELS), dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)

    if 'MNIST' in args.dataset:
        agent_model = model_mnist(type=args.model_num)
    elif args.dataset == 'census':
        agent_model = census_model_1()
    elif 'kather' in args.dataset:
        agent_model = model_kather()
    elif 'CIFAR' in args.dataset:
        agent_model = model_cifar()

    logits = agent_model(x)

    if args.dataset == 'census':
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    else:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    elif args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        sess = tf.Session(config=config)
    elif args.k == 1:
        sess = tf.Session()

    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    agent_model.set_weights(shared_weights)

    start_offset = 0
    if args.steps is not None:
        start_offset = (t * args.B * args.steps) % (data_size - args.B)

    for e in range(args.E):
        X_batch = X
        Y_batch = Y
        Y_batch_uncat = Y
        _, loss_val = sess.run([optimizer, loss], feed_dict={x: X_batch, y: Y_batch_uncat})

    local_weights = agent_model.get_weights()
    local_delta = local_weights - shared_weights

    eval_success, eval_loss = eval_minimal(X, Y, shared_weights+local_delta)
    print('success {}, loss {}'.format(eval_success, eval_loss))

    clean_model_weights = np.array(local_delta)

    return clean_model_weights


def master():
    K.set_learning_phase(1)

    args = gv.args
    print('Initializing master model')
    config = tf.ConfigProto(gpu_options=gv.gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    if 'MNIST' in args.dataset:
        global_model = model_mnist(type=args.model_num)
    elif 'census' in args.dataset:
        global_model = census_model_1()
    elif 'kather' in args.dataset:
        global_model = model_kather()
    elif 'CIFAR' in args.dataset:
        global_model = model_cifar()

    global_model.summary()
    global_weights_np = global_model.get_weights()
    np.save(gv.dir_name + 'global_weights_t0.npy', global_weights_np)

    return
