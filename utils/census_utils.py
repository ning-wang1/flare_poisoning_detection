#########################
# Purpose: Utility functions for attacks on the census data
########################

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout
import scipy.io as sio
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from keras.utils import np_utils

import global_vars as gv
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

inputs = (
        ("age", ("continuous",)),
        ("workclass", (
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay",
            "Never-worked")),
        ("fnlwgt", ("continuous",)),
        ("education", (
            "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th",
            "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")),
        ("education-num", ("continuous",)),
        ("marital-status", (
            "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent",
            "Married-AF-spouse")),
        ("occupation", ("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                        "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                        "Priv-house-serv", "Protective-serv", "Armed-Forces")),
        ("relationship", ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried")),
        ("race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")),
        ("sex", ("Female", "Male")),
        ("capital-gain", ("continuous",)),
        ("capital-loss", ("continuous",)),
        ("hours-per-week", ("continuous",)),
        ("native-country", (
            "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
            "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland",
            "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
            "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
            "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"))
    )


def get_input_dim():

    input_shape = []
    for i in inputs:
        count = len(i[1 ])
        input_shape.append(count)
    input_dim = sum(input_shape)
    print("input_shape:", input_shape)
    print("input_dim:", input_dim)
    print()
    outputs = (0, 1)  # (">50K", "<=50K")
    output_dim = 2  # len(outputs)
    print("output_dim:", output_dim)
    print()
    return input_shape


def isFloat(string):
    #  http://stackoverflow.com/questions/2356925/how-to-check-whether-string-might-be-type-cast-to-float-in-python
    try:
        float(string)
        return True
    except ValueError:
        return False


def find_means_for_continuous_types(X):
    means = []
    for col in range(len(X[0])):
        summ = 0
        count = 0.000000000000000000001
        for value in X[:, col]:
            if isFloat(value):
                summ += float(value)
                count += 1
        means.append(summ / count)
    return means


def flatten_persons_inputs_for_model(person_inputs, input_shape, means):

    float_inputs = []

    for i in range(len(input_shape)):
        features_of_this_type = input_shape[i]
        is_feature_continuous = features_of_this_type == 1

        if is_feature_continuous:
            mean = means[i]
            if isFloat(person_inputs[i]):
                scale_factor = 1 / (2 * mean)  # we prefer inputs mainly scaled from -1 to 1.
                float_inputs.append(float(person_inputs[i]) * scale_factor)
            else:
                float_inputs.append(mean)
        else:
            for j in range(features_of_this_type):
                feature_name = inputs[i][1][j]

                if feature_name == person_inputs[i]:
                    float_inputs.append(1.)
                else:
                    float_inputs.append(0)
    return float_inputs


def prepare_data(raw_data, input_shape, means):
    X = raw_data[:, :-1]
    y = raw_data[:, -1:]

    # X:
    new_X = []
    for person in range(len(X)):
        formatted_X = flatten_persons_inputs_for_model(X[person], input_shape, means)
        new_X.append(formatted_X)
    new_X = np.array(new_X)

    # y:
    new_y = []
    for i in range(len(y)):
        if ">50K" in y[i] or ">50K." in y[i]:
            new_y.append((0, 1))
        else:  # y[i] == "<=50k":
            new_y.append((1, 0))
    new_y = np.array(new_y)

    return new_X, new_y


def data_census():

    # Building training and test data
    data_dir = 'data/census/'
    training_data = np.genfromtxt(data_dir + 'adult.data', delimiter=', ', dtype=str, autostrip=True)
    print("Training data count:", len(training_data))
    test_data = np.genfromtxt(data_dir+'adult.test', delimiter=', ', dtype=str, autostrip=True)
    print("Test data count:", len(test_data))

    means = find_means_for_continuous_types(np.concatenate((training_data, test_data), 0))
    print("Mean values for data types (if continuous):", means)
    input_shape = get_input_dim()

    X_train, y_train = prepare_data(training_data, input_shape, means)
    X_test, y_test = prepare_data(test_data, input_shape, means)
    X_train, y_train = X_train[0:32000], y_train[0:32000]
    return X_train, y_train, X_test, y_test


def census_model_1():
    main_input = Input(shape=(gv.DATA_DIM,), name='main_input')
    x = Dense(256, use_bias=True, activation='relu')(main_input)
    x = Dropout(0.5)(x)
    x = Dense(256, use_bias=True, activation='relu')(x)
    x = Dropout(0.5)(x)
    # main_output = Dense(1)(x)
    main_output = Dense(gv.NUM_CLASSES)(x)

    model = Model(inputs=main_input, outputs=main_output)

    return model


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = data_census()
