import os
from PIL import Image
import numpy as np

new_width = 64
new_height = 64
old_direct = 'Kather_texture_2016_image_tiles_5000'
new_direct_train = 'Kather_resize64/train'
new_direct_test = 'Kather_resize64/test'


def transform(filepath, train_set):
    img = Image.open(filepath) # image extension *.png,*.jpg
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    if train_set:
        new_filepath = filepath.replace(old_direct, new_direct_train)
    else:
        new_filepath = filepath.replace(old_direct, new_direct_test)
    img.save(new_filepath)


def resize_images():
    solupath = []
    dir_cur = os.getcwd()
    path = dir_cur+'/Kather_texture_2016_image_tiles_5000'
    directories = [x[0] for x in os.walk(path)]
    # directories[0] is main dir，[1:] are sub-dir for classes

    for label, directory in enumerate(directories[1:]):
        class_num = [os.path.join(label) for label in os.listdir(directory)]

        all_path = [os.path.join(directory, label) for label in class_num]
        solupath.extend(all_path)

    total_num = len(solupath)
    split = int(total_num * 0.8)
    indices = np.arange(total_num)
    np.random.shuffle(indices)
    train_idx, test_idx = indices[:split], indices[split:]

    for idx, img in enumerate(solupath):
        if idx % 100 == 0:
            print('saving {} / {}'.format(idx, total_num))
        train_set = idx in train_idx
        transform(img, train_set)
    print('loading data')


if __name__ == "__main__":
    print('start')
    resize_images()


# import os
# import imageio
# from PIL import Image

# import tensorflow.keras.backend as K
# import numpy as np
# import tensorflow.compat.v1 as tf
# import global_vars as gv

# from utils.kather_utils import model_kather, data_kather

# from keras.models import Sequential, model_from_json
# from keras.layers import Dense, Dropout, Activation, Flatten, Input
# from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
# from keras.utils import np_utils
# import copy
# import warnings
# warnings.filterwarnings('ignore')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


