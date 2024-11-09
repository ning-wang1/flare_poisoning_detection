#########################
# Purpose: Utility functions for attacks on the cancer data
########################

import os
import imageio
from PIL import Image

import keras.backend as K
import numpy as np
import tensorflow.compat.v1 as tf
import global_vars as gv

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
import copy
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def transform(filename):
    image = imageio.imread(filename)
    if len(image.shape) < 3:
        image = np.array([image for i in range(3)])
    else:
        image = image

    return np.array(image)


def read_images(path):
    solupath = []
    directories = [x[0] for x in os.walk(path)]
    # directories[0] is main dir，[1:] are sub-dir for classes
    labels = []
    num = 0
    for label, directory in enumerate(directories[1:]):
        class_num = [os.path.join(label) for label in os.listdir(directory)]

        all_path = [os.path.join(directory, label) for label in class_num]
        solupath.extend(all_path)
        for i, j in enumerate(all_path):
            labels.append(num)
        num += 1
    images = np.array([transform(img) for img in solupath])
    print('loading data')
    return images, np.array(labels)


def data_kather():
    # path = os.path.abspath(os.path.dirname(os.getcwd()))
    data_dir_train = 'data/Kather_resized/train'
    data_dir_test = 'data/Kather_resized/test'

    X_train, y_train = read_images(data_dir_train)
    total_num = X_train.shape[0]
    indices = np.arange(total_num)
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]
    X_test, y_test = read_images(data_dir_test)

    X_train = X_train.reshape(X_train.shape[0],
                              gv.IMAGE_ROWS,
                              gv.IMAGE_COLS,
                              gv.NUM_CHANNELS)

    X_test = X_test.reshape(X_test.shape[0],
                            gv.IMAGE_ROWS,
                            gv.IMAGE_COLS,
                            gv.NUM_CHANNELS)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, gv.NUM_CLASSES).astype(np.float32)
    y_test = np_utils.to_categorical(y_test, gv.NUM_CLASSES).astype(np.float32)

    return X_train, y_train, X_test, y_test


def model_kather():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(gv.IMAGE_ROWS,
                                                               gv.IMAGE_COLS,
                                                               gv.NUM_CHANNELS), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))  #128
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))  # 128
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.3))

    model.add(Dense(gv.NUM_CLASSES))
    return model


def model_kather_0():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(gv.IMAGE_ROWS,
                                                               gv.IMAGE_COLS,
                                                               gv.NUM_CHANNELS), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))  #128
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(gv.NUM_CLASSES))
    return model


def model_kather_2():
    print('model for kather data')
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(gv.IMAGE_ROWS,
                                                               gv.IMAGE_COLS,
                                                               gv.NUM_CHANNELS),
                     activation="relu"))

    model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
    model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))
    model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))
    model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(gv.NUM_CLASSES))
    return model


if __name__ == '__main__':
    gv.init()
    tf.set_random_seed(777)
    np.random.seed(777)
    args = gv.args
    X_train, y_train, X_test, y_test = data_kather()
    print('Training image shape: {}, and the number of instances are: {}'.format(X_train.shape, y_train.shape))
    print('Testing image shape: {}, and the number of instances are: {}'.format(X_test.shape, y_test.shape))
    print('the end')



# def data_kather():
#     # path = os.path.abspath(os.path.dirname(os.getcwd()))
#     data_dir ='data/Kather_texture_2016_image_tiles_5000'
#     x, y = read_images(data_dir)
#     total_num = x.shape[0]
#     split = int(total_num * 0.8)
#     indices = np.arange(total_num)
#
#     np.random.shuffle(indices)
#     train_idx, test_idx = indices[:split], indices[split:]
#     X_train, y_train = x[train_idx], y[train_idx]
#     X_test, y_test = x[test_idx], y[test_idx]
#
#     X_train = X_train.reshape(X_train.shape[0],
#                               gv.IMAGE_ROWS,
#                               gv.IMAGE_COLS,
#                               gv.NUM_CHANNELS)
#
#     X_test = X_test.reshape(X_test.shape[0],
#                             gv.IMAGE_ROWS,
#                             gv.IMAGE_COLS,
#                             gv.NUM_CHANNELS)
#
#     X_train = X_train.astype('float32')
#     X_test = X_test.astype('float32')
#     X_train /= 255
#     X_test /= 255
#     print('X_train shape:', X_train.shape)
#     print(X_train.shape[0], 'train samples')
#     print(X_test.shape[0], 'test samples')
#
#     # convert class vectors to binary class matrices
#     y_train = np_utils.to_categorical(y_train, gv.NUM_CLASSES).astype(np.float32)
#     y_test = np_utils.to_categorical(y_test, gv.NUM_CLASSES).astype(np.float32)
#
#     return X_train, y_train, X_test, y_test