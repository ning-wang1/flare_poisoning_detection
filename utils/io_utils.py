#########################
# Purpose: Help with file input/output
########################
import os
import global_vars as gv
import numpy as np

from utils.mnist import data_mnist
from keras.datasets import cifar10
from keras.utils import np_utils
from utils.fmnist import load_fmnist
from utils.kather_utils import data_kather
from utils.cifar_utils import data_cifar
from utils.census_utils import data_census
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def file_write(write_dict, purpose='global_eval_loss'):
    f = open(gv.output_dir_name + gv.output_file_name +
             '_' + purpose + '.txt', 'a')
    if write_dict['t'] == 1:
        d_count = 1
        for k, v in write_dict.items():
            if d_count < len(write_dict):
                f.write(k + ',')
            else:
                f.write(k + '\n')
            d_count += 1
        d_count = 1
        for k, v in write_dict.items():
            if d_count < len(write_dict):
                f.write(str(v) + ',')
            else:
                f.write(str(v) + '\n')
            d_count += 1
    elif write_dict['t'] != 1:
        d_count = 1
        for k, v in write_dict.items():
            if d_count < len(write_dict):
                f.write(str(v) + ',')
            else:
                f.write(str(v) + '\n')
            d_count += 1
    f.close()


def data_setup():
    args = gv.args
    if 'MNIST' in args.dataset:
        X_train, Y_train, X_test, Y_test = data_mnist()
        Y_test_uncat = np.argmax(Y_test, axis=1)
        Y_train_uncat = np.argmax(Y_train, axis=1)
        print('Loaded f/MNIST data')
    elif args.dataset == 'CIFAR-10':
        X_train, Y_train, X_test, Y_test = data_cifar()
        Y_test_uncat = np.argmax(Y_test, axis=1)
        Y_train_uncat = np.argmax(Y_train, axis=1)

        print('Loaded CIFAR-10 data')
    elif args.dataset == 'census':
        X_train, Y_train, X_test, Y_test = data_census()
        Y_test_uncat = np.argmax(Y_test, axis=1)
        Y_train_uncat = np.argmax(Y_train, axis=1)
        print(Y_test)
        print(Y_test_uncat)
        print('Loaded Census data')
    elif 'kather' in args.dataset:
        X_train, Y_train, X_test, Y_test = data_kather()
        Y_test_uncat = np.argmax(Y_test, axis=1)
        Y_train_uncat = np.argmax(Y_train, axis=1)
        print('Loaded Kather data')
    return X_train, Y_train, Y_train_uncat, X_test, Y_test, Y_test_uncat


def mal_data_create(X_test, Y_test, Y_test_uncat, mal_num=20, original_class=4, target_class=3):
    args = gv.args

    if args.mal_obj == 'all':
        mal_data_X = X_test
        mal_data_Y = Y_test
        true_labels = Y_test
    elif args.mal_obj == 'single':
        r = np.random.choice(len(X_test))
        if original_class is not None:
            while Y_test_uncat[r] != original_class:
                r = np.random.choice(len(X_test))
            print(r)
        if 'semantic' in args.trojan:
            mal_data_X = X_test[r:r + 1]
        elif 'trojan' in args.trojan:
            mal_data_X = poison(X_test[r:r+1], 'pattern', [gv.IMAGE_ROWS-2, gv.IMAGE_COLS-2], 255)
        else:
            print('please select trojan or semantic as your backdoor goal')
        allowed_targets = list(range(gv.NUM_CLASSES))
        print("Initial class: %s" % Y_test_uncat[r])
        true_labels = Y_test_uncat[r:r + 1]
        if target_class is None:
            allowed_targets.remove(Y_test_uncat[r])
            mal_data_Y = np.random.choice(allowed_targets)
        else:
            mal_data_Y = target_class

        mal_data_Y = np.array(mal_data_Y).reshape(1, )
        print("Target class: %s" % mal_data_Y[0])
    elif 'multiple' in args.mal_obj:
        target_indices = np.random.choice(len(X_test), mal_num)
        mal_data_X = X_test[target_indices]
        print("Initial classes: %s" % Y_test_uncat[target_indices])
        true_labels = Y_test_uncat[target_indices]
        mal_data_Y = []
        for i in range(mal_num):
            allowed_targets = list(range(gv.NUM_CLASSES))
            allowed_targets.remove(Y_test_uncat[target_indices[i]])
            mal_data_Y.append(np.random.choice(allowed_targets))
        mal_data_Y = np.array(mal_data_Y)
    elif 'target_backdoor' in args.mal_obj:
        target_indices = np.random.choice(len(X_test), mal_num)
        mal_data_X = X_test[target_indices]
        mal_class = np.random.choice(gv.NUM_CLASSES)
        mal_data_X = poison(mal_data_X, 'pattern', [gv.IMAGE_ROWS - 2, gv.IMAGE_COLS - 2], 255)
        true_labels = Y_test_uncat[target_indices]
        mal_data_Y = []
        for i in range(mal_num):
            mal_data_Y.append(mal_class)
        mal_data_Y = np.array(mal_data_Y)

    with open('data/mal_X_%s_%s_%s' % (args.dataset, args.mal_obj, args.trojan), 'wb') as f:
        np.save(f, mal_data_X)
    with open('data/mal_Y_%s_%s_%s.npy' % (args.dataset, args.mal_obj, args.trojan), 'wb') as f:
        np.save(f, mal_data_Y)
    with open('data/true_labels_%s_%s_%s.npy' % (args.dataset, args.mal_obj, args.trojan), 'wb') as f:
        np.save(f, true_labels)

    return mal_data_X, mal_data_Y, true_labels


def poison(X_test, method, pos, col):
    ret_x = np.copy(X_test)
    col_arr = np.asarray(col)
    if ret_x.ndim == 3:
        # only one image was passed
        if method == 'pixel':
            ret_x[pos[0], pos[1], :] = col_arr
        elif method == 'pattern':
            ret_x[pos[0], pos[1], :] = col_arr
            ret_x[pos[0] + 1, pos[1] + 1, :] = col_arr
            ret_x[pos[0] - 1, pos[1] + 1, :] = col_arr
            ret_x[pos[0] + 1, pos[1] - 1, :] = col_arr
            ret_x[pos[0] - 1, pos[1] - 1, :] = col_arr
        elif method == 'ell':
            ret_x[pos[0], pos[1], :] = col_arr
            ret_x[pos[0] + 1, pos[1], :] = col_arr
            ret_x[pos[0], pos[1] + 1, :] = col_arr
    elif ret_x.ndim == 4:
        # batch was passed
        if method == 'pixel':
            ret_x[:, pos[0], pos[1], :] = col_arr
        elif method == 'pattern':
            ret_x[:, pos[0], pos[1], :] = col_arr
            ret_x[:, pos[0] + 1, pos[1] + 1, :] = col_arr
            ret_x[:, pos[0] - 1, pos[1] + 1, :] = col_arr
            ret_x[:, pos[0] + 1, pos[1] - 1, :] = col_arr
            ret_x[:, pos[0] - 1, pos[1] - 1, :] = col_arr
        elif method == 'ell':
            ret_x[:, pos[0], pos[1], :] = col_arr
            ret_x[:, pos[0] + 1, pos[1], :] = col_arr
            ret_x[:, pos[0], pos[1] + 1, :] = col_arr
    else:
        print('input data dimensionality error')
    return ret_x


def mal_data_setup(X_test, Y_test, Y_test_uncat, mal_num, gen_flag=True):
    args = gv.args

    data_path = 'data/mal_X_%s_%s.npy' % (args.dataset, args.mal_obj)
    print(data_path)

    if os.path.exists('data/mal_X_%s_%s_%s.npy' % (args.dataset, args.mal_obj, args.trojan)):
        mal_data_X = np.load('data/mal_X_%s_%s_%s.npy' % (args.dataset, args.mal_obj, args.trojan))
        mal_data_Y = np.load('data/mal_Y_%s_%s_%s.npy' % (args.dataset, args.mal_obj, args.trojan))
        true_labels = np.load('data/true_labels_%s_%s_%s.npy' % (args.dataset, args.mal_obj, args.trojan))
    else:
        if gen_flag:
            if 'census' in args.dataset:
                mal_data_X, mal_data_Y, true_labels = mal_data_create(X_test, Y_test, Y_test_uncat, mal_num,
                                                                      original_class=0, target_class=1)
            else:
                mal_data_X, mal_data_Y, true_labels = mal_data_create(X_test, Y_test, Y_test_uncat, mal_num)
        else:
            raise ValueError('Tried to generate data but not allowed')

    print("Initial classes: %s" % true_labels)
    print("Target classes: %s, malicious data shape: %s" % (mal_data_Y, mal_data_Y.shape))

    return mal_data_X, mal_data_Y, true_labels
