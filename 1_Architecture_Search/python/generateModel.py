import keras
from keras.utils.layer_utils import count_params
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
from scipy.stats import norm
import random
import tensorflow as tf
import os
import math
from copy import copy
from sklearn.model_selection import StratifiedKFold
from kerascosineannealing.cosine_annealing import CosineAnnealingScheduler
import constants
import custom_early_stopping_callback

from utils_gpu import pick_gpu_lowest_memory

os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())


def get_model_withParam(solution_archi, inputShape):
    print("ENTER NEW GENERATE MODEL")
    decoder_nb_layer = len(solution_archi)
    # print(decoder_nb_layer)
    # print("***")
    # for i in range(decoder_nb_layer):
    #    number_of_convlayer = len(solution_archi[i])-1
    #    print("--"+str(i))
    #    print("----"+str(number_of_convlayer))
    #    print("--**")
    #    for j in range(number_of_convlayer):
    #        print("----"+str(solution_archi[i][j]))
    #    print("----"+str(solution_archi[i][number_of_convlayer]))

    model = Sequential()

    first = False

    decoder_nb_layer = len(solution_archi)

    for i in range(decoder_nb_layer):
        number_of_convlayer = len(solution_archi[i]) - 1
        for j in range(number_of_convlayer):

            nb_kernel = solution_archi[i][j][0]
            kernel_size = solution_archi[i][j][1]
            stride = solution_archi[i][j][2]

            if not first:
                model.add(Conv2D(nb_kernel, (kernel_size, kernel_size), strides=(stride, stride), padding='same',
                                 kernel_regularizer=regularizers.l2(constants.weight_decay), input_shape=inputShape))
                first = True
            else:
                model.add(Conv2D(nb_kernel, (kernel_size, kernel_size), strides=(stride, stride), padding='same',
                                 kernel_regularizer=regularizers.l2(constants.weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

        maxPoolSize = solution_archi[i][number_of_convlayer]
        model.add(MaxPooling2D(pool_size=(maxPoolSize, maxPoolSize)))
        model.add(Dropout(i / (2 * decoder_nb_layer)))
        print((i + 1) / (2 * decoder_nb_layer))

    print("**")
    model.add(Dropout(0.20))
    model.add(BatchNormalization())
    model.add(Dense(2000))
    model.add(Flatten())
    model.add(Dense(constants.num_classes, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


datagen_1 = ImageDataGenerator(
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=5,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=5,
    channel_shift_range=0.1,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    validation_split=0.0)

datagen_2 = copy(datagen_1)


def unison_shuffling_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))

    return a[p], b[p]


def mixing_generator_2(generator_1):
    while True:
        (batch_x_1, batch_y_1) = next(generator_1)
        (batch_x_2, batch_y_2) = unison_shuffling_copies(batch_x_1, batch_y_1)

        # alpha = 1
        alpha = 0.4

        lam = np.random.beta(alpha, alpha)
        batch_x = (lam * batch_x_1 + (1. - lam) * batch_x_2)
        batch_y = (lam * batch_y_1 + (1. - lam) * batch_y_2)

        yield batch_x, batch_y


def fit_normal_distribution_1d(data_1d):
    mu, std = norm.fit(data_1d)
    return mu, std


def fit_normal_distribution_2d(data_2d):
    if not isinstance(data_2d, np.ndarray):
        data_2d = np.vstack(data_2d)

    mu_std_list = [norm.fit(data_2d[:, i]) for i in range(data_2d.shape[-1])]
    return mu_std_list


def update_top_N(top_N, val_acc_history, archi_idx, solution_archi):
    is_updated = False
    if len(top_N['top_N_val_acc']) < constants.top_N:
        top_N['top_N_val_acc'].append(val_acc_history)
        top_N['top_N_arch_id'].append(archi_idx)
        top_N['top_N_solution_arch'].append(solution_archi)
        is_updated = True

        if len(top_N['top_N_val_acc']) == constants.top_N:
            top_N['top_N_mean_std'] = fit_normal_distribution_2d(top_N['top_N_val_acc'])
    else:
        if (np.max(val_acc_history) >= np.min([np.max(acc) for acc in top_N['top_N_val_acc']])):
            idx_min = np.argmin([np.max(acc) for acc in top_N['top_N_val_acc']])
            if len(val_acc_history) < constants.max_epoch:
                val_acc_history = val_acc_history + [np.max(val_acc_history)]*(constants.max_epoch - len(val_acc_history))
            top_N['top_N_val_acc'] = top_N['top_N_val_acc'][:idx_min] + top_N['top_N_val_acc'][idx_min + 1:] + [val_acc_history]
            top_N['top_N_mean_std'] = fit_normal_distribution_2d(top_N['top_N_val_acc'] )
            top_N['top_N_arch_id'] = top_N['top_N_arch_id'][:idx_min] + top_N['top_N_arch_id'][idx_min + 1:] + [archi_idx]
            top_N['top_N_solution_arch'] = top_N['top_N_solution_arch'][:idx_min] + top_N['top_N_solution_arch'][idx_min + 1:] + [solution_archi]
            is_updated = True

    return top_N, is_updated


def trainWithKFOld(solution_archi, train_data, train_labels, archi_idx, top_N):
    print("ENTER trainWithKFOld")
    # train_labels_withoutKfold = np_utils.to_categorical(train_labels, constants.num_classes)

    os.environ['PYTHONHASHSEED'] = str(constants.seed_value)
    random.seed(constants.seed_value)
    np.random.seed(constants.seed_value)
    if tf.__version__[0] == "1":
        tf.set_random_seed(constants.seed_value)  # TF 1
        config = tf.ConfigProto(allow_soft_placement=True)  # TF 1
    elif tf.__version__[0] == "2":
        tf.random.set_seed(constants.seed_value)  # TF 2
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # TF 2

    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True

    if tf.__version__[0] == "1":
        sess = tf.Session(config=config)  # TF 1
        keras.backend.set_session(sess)  # TF 1
    elif tf.__version__[0] == "2":
        sess = tf.compat.v1.Session(config=config)  # TF 2
        tf.compat.v1.keras.backend.set_session(sess)  # TF 2

    # Allow memory growth for the GPU
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    skf = StratifiedKFold(n_splits=constants.K_fold, shuffle=False)

    # model.summary()

    nb_run = 1
    for index, (train_indices, val_indices) in enumerate(skf.split(train_data, train_labels)):


        # print("K FOLD OF " + str(index))
        xtrain, xtest = train_data[train_indices], train_data[val_indices]
        ytrain, ytest = train_labels[train_indices], train_labels[val_indices]

        ytrain = np_utils.to_categorical(ytrain, constants.num_classes)
        ytest = np_utils.to_categorical(ytest, constants.num_classes)

        callbacks_list = []
        # Learning rate scheduler callback definition
        callbacks_list.append(CosineAnnealingScheduler(T_max=constants.max_epoch, eta_max=constants.new_lr, eta_min=1e-5))

        # CSV logger callback definition
        csv_logger_output_filename = os.path.join(constants.csv_logger_output_folder,
                                                  "arch_{}_run_{}.csv".format(archi_idx, nb_run))
        callbacks_list.append(keras.callbacks.CSVLogger(csv_logger_output_filename, separator=';', append=True))

        # Early Stopping callback definition
        if top_N["top_N_mean_std"] != []:
            percentile_threshold = constants.percentile_threshold
            callbacks_list.append(custom_early_stopping_callback.CustomEarlyStoppingCallback(top_N["top_N_mean_std"],
                                                                                             percentile_threshold))

        datagen_1.fit(xtrain)
        datagen_1_flow = datagen_1.flow(xtrain, ytrain, batch_size=constants.batch_size)
        datagen_flow = mixing_generator_2(datagen_1_flow)

        steps = math.ceil(len(ytrain) / constants.batch_size)

        # TEST
        model = get_model_withParam(solution_archi, train_data.shape[1:])

        history = model.fit_generator(datagen_flow,
                                      epochs=constants.max_epoch,
                                      validation_data=(xtest, ytest),
                                      steps_per_epoch=steps,
                                      callbacks=callbacks_list)

        val_acc_history = history.history[constants.val_acc_name]
        top_acc = np.max(val_acc_history)

        top_N, is_updated = update_top_N(top_N, val_acc_history, archi_idx, solution_archi)

        nb_run += 1
        if nb_run > constants.max_nb_runs:
            break

    # print(np.mean(all_scores))
    if tf.__version__[0] == "1":
        keras.backend.clear_session()  # TF 1
    elif tf.__version__[0] == "2":
        tf.keras.backend.clear_session()  # TF 2

    trainable_count = count_params(model.trainable_weights)
    # non_trainable_count = count_params(model.non_trainable_weights)

    print("EXIT trainWithKFOld")
    return top_N, is_updated, top_acc, trainable_count


def ParaFunction(solution_archi, solution_optim, train_data, train_labels, test_data, test_labes, F):
    F.value = trainWithoutKFOld(solution_archi, train_data, train_labels, test_data, test_labes)


def ParaFunctionWitKFold(solution_archi, solution_optim, train_data, train_labels, F):
    F.value = np.mean(trainWithKFOld(solution_archi, train_data, train_labels))
