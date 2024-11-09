#########################
# Purpose: Useful functions for evaluating a model on test data
########################
import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
# tf.set_random_seed(777)
# np.random.seed(777)
import keras.backend as K
import copy
from keras.utils import np_utils
from keras.models import Model
from utils.mnist import model_mnist
from utils.census_utils import census_model_1
from utils.kather_utils import model_kather
from utils.cifar_utils import model_cifar
import global_vars as gv
from utils.io_utils import file_write
from collections import OrderedDict
import os

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def eval_setup(global_weights):
    args = gv.args
    global_weights_np = global_weights
    if 'MNIST' in args.dataset:
        K.set_learning_phase(0)

    # input
    if args.dataset == 'census':
        x = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32)
    else:
        x = tf.placeholder(shape=(None, gv.IMAGE_ROWS, gv.IMAGE_COLS, gv.NUM_CHANNELS), dtype=tf.float32)
    y = tf.placeholder(dtype=tf.int64)

    # model
    if 'MNIST' in args.dataset:
        global_model = model_mnist(type=args.model_num)
        layer_idx = 7
    elif 'CIFAR-10' in args.dataset:
        global_model = model_cifar()
        layer_idx = 7
    elif 'census' in args.dataset:
        global_model = census_model_1()
        layer_idx = 3
    elif 'kather' in args.dataset:
        global_model = model_kather()
        layer_idx = 11#13

    # layer = global_model.layers[layer_idx]  # the penultimate layer
    layer_model = Model(global_model.inputs, global_model.layers[layer_idx].output)
    layer_output = layer_model(x)

    logits = global_model(x)
    prediction = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        # config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    elif args.k == 1:
        sess = tf.Session()
    
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    global_model.set_weights(global_weights_np)

    return x, y, sess, prediction, layer_output, loss


def mal_eval_single(mal_data_X, mal_data_Y, global_weights):
    """
    return the target and true prediction for malicious data
    :param mal_data_X:
    :param mal_data_Y:
    :param global_weights:
    :return:
    """

    x, y, sess, prediction, layer_output, loss = eval_setup(global_weights)

    mal_obj_pred = sess.run(prediction, feed_dict={x: mal_data_X})
    target = mal_data_Y[0]
    target_conf = mal_obj_pred[:, mal_data_Y][0][0]
    actual = np.argmax(mal_obj_pred, axis=1)[0]
    actual_conf = np.max(mal_obj_pred, axis=1)[0]

    sess.close()

    return target, target_conf, actual, actual_conf


def mal_eval_multiple(mal_data_X, mal_data_Y, global_weights):
    """
    evaluate the model on malicious data. The adversary has multiple input instances
    :param mal_data_X:
    :param mal_data_Y:
    :param global_weights:
    :return:
    """

    x, y, sess, prediction, layer_output, loss = eval_setup(global_weights)

    mal_obj_pred = sess.run(prediction, feed_dict={x: mal_data_X})
    print('obj.......................', np.argmax(mal_obj_pred, axis=1))
    suc_count_local = np.sum(mal_data_Y==np.argmax(mal_obj_pred, axis=1))

    return suc_count_local


def eval_minimal(X_test, Y_test, global_weights, return_dict=None):
    """
    evaluate the a model
    :param X_test:
    :param Y_test:
    :param global_weights:
    :param return_dict:
    :return:
    """

    args = gv.args

    x, y, sess, prediction, layer_output, loss = eval_setup(global_weights)

    pred_np = np.zeros((len(X_test), gv.NUM_CLASSES))
    eval_loss = 0.0

    if args.dataset == 'CIFAR-10' or args.dataset == 'kather':
        Y_test = Y_test.reshape(len(Y_test))

    for i in range(int(len(X_test) / gv.BATCH_SIZE)):
        X_test_slice = X_test[i * (gv.BATCH_SIZE):(i + 1) * (gv.BATCH_SIZE)]
        Y_test_slice = Y_test[i * (gv.BATCH_SIZE):(i + 1) * (gv.BATCH_SIZE)]
        # Y_test_cat_slice = np_utils.to_categorical(Y_test_slice)
        pred_np_i = sess.run(prediction, feed_dict={x: X_test_slice})
        eval_loss += sess.run(loss, feed_dict={x: X_test_slice, y: Y_test_slice})
        pred_np[i * gv.BATCH_SIZE:(i + 1) * gv.BATCH_SIZE, :] = pred_np_i
    eval_loss = eval_loss / (len(X_test) / gv.BATCH_SIZE)

    if args.dataset == 'CIFAR-10' or args.dataset == 'kather':
        Y_test = Y_test.reshape(len(Y_test))
    eval_success = 100.0 * np.sum(np.argmax(pred_np, 1) == Y_test) / len(Y_test)

    sess.close()

    if return_dict is not None:
        return_dict['success_thresh'] = eval_success

    return eval_success, eval_loss


def eval_func(X_test, Y_test, t, return_dict, mal_data_X=None, mal_data_Y=None, global_weights=None):
    """
    evaluate the model and write the result to file
    :param X_test:
    :param Y_test:
    :param t:
    :param return_dict:
    :param mal_data_X:
    :param mal_data_Y:
    :param global_weights:
    :return:
    """
    args = gv.args 

    # if global_weights is None:
    #     global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t)

    if args.dataset == 'CIFAR-10' or args.dataset == 'kather':
        K.set_learning_phase(1)
    eval_success, eval_loss = eval_minimal(X_test, Y_test, global_weights)

    print('Results------>Iteration {}: success {}, loss {}'.format(t, eval_success, eval_loss))
    write_dict = OrderedDict()
    write_dict['t'] = t
    write_dict['eval_success'] = eval_success
    write_dict['eval_loss'] = eval_loss
    file_write(write_dict)

    return_dict['eval_success'] = eval_success
    return_dict['eval_loss'] = eval_loss
    # print(global_weights[9][0:5])
    if args.mal and 'backdoor' in args.attack_type:
        if 'single' in args.mal_obj:
            target, target_conf, actual, actual_conf = mal_eval_single(mal_data_X, mal_data_Y, global_weights)
            print('------------->Target:%s with conf. %s, Curr_pred on for iter %s:%s with conf. %s' % (
                target, target_conf, t, actual, actual_conf))
            if actual == target:
                return_dict['mal_suc_count'] += 1
                return_dict['attack_succeed'] = 1
            else:
                return_dict['attack_succeed'] = 0
            write_dict = OrderedDict()
            write_dict['t'] = t
            write_dict['target'] = target
            write_dict['target_conf'] = target_conf
            write_dict['actual'] = actual
            write_dict['actual_conf'] = actual_conf
            file_write(write_dict, purpose='mal_obj_log')
            return_dict['target_conf'] = target_conf
        elif 'multiple' in args.mal_obj or 'target_backdoor' in args.mal_obj:
            suc_count_local = mal_eval_multiple(mal_data_X, mal_data_Y, global_weights)    
            print('%s of %s targets achieved' % (suc_count_local, args.mal_num))
            write_dict = OrderedDict()
            write_dict['t'] = t
            write_dict['suc_count'] = suc_count_local
            file_write(write_dict, purpose='mal_obj_log')
            return_dict['mal_suc_count'] += suc_count_local

    return


def compute_roc(probs_neg, probs_pos, plot=False):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.style.use('seaborn-dark')
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        # plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score


def compute_roc_1(y, scores, plot=False):
    """
    TODO
    :param y:
    :param scores:
    :param plot:
    :return:
    """

    fpr, tpr, thresholds = roc_curve(y, scores)
    auc_score = auc(fpr, tpr)
    if plot:
        pp = PdfPages('test1.pdf')
        # plt.style.use('seaborn-dark')
        plt.figure(figsize=(6, 4.5))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        # plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.grid(linestyle='--', linewidth=1)
        # plt.show()
        pp.savefig()
        plt.close()
        pp.close()
    return fpr, tpr, auc_score


