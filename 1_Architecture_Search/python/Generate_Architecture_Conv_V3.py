import keras
from keras.datasets import cifar10, cifar100, mnist
import numpy as np
from keras.utils import np_utils
import os
import random
import tensorflow as tf

import generateModel as GM
# import run_V2 as run
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import multiprocessing

import pickle
import constants

import utils_slack
import cv2
import shutil


def generate_new_architecture():
    number_of_layers = [2, 3, 4]
    number_of_conv = [2, 3, 4]

    nb_kernel = [8]
    kernel_size = [2, 3, 4, 5]
    stride = [1]

    maxPooling_size = [2]

    number_of_layers_choice = random.choice(number_of_layers)
    final_result = [None] * number_of_layers_choice

    # print(number_of_layers_choice)
    # print("***")

    for i in range(number_of_layers_choice):
        # print("--"+str(i))
        number_of_conv_choice = random.choice(number_of_conv)
        # print("----"+str(number_of_conv_choice))
        final_result[i] = []

        # print("--**")

        for j in range(number_of_conv_choice):

            nb_kernel_choice = random.choice(nb_kernel)
            kernel_size_choice = random.choice(kernel_size)
            stride_choice = random.choice(stride)

            final_result[i].append([2**nb_kernel_choice, kernel_size_choice, stride_choice])
            
            # print("----"+str([2**nb_kernel_choice,kernel_size_choice,stride_choice]))
        maxPooling_size_choice = random.choice(maxPooling_size)
        final_result[i].append(maxPooling_size_choice)
        
        # print("----"+str([maxPooling_size_choice,maxPooling_size_choice]))

    # print("***")
    # print(final_result)
    # print("****************************************************")
    return final_result


def main():
    # READ DATA FRAME
    if constants.dataset_type in ["MNIST"]:
        (full_x_train, full_y_train), (full_x_test, full_y_test) = mnist.load_data()
        full_x_train = full_x_train.reshape((full_x_train.shape[0], 28, 28, 1))
        full_x_test = full_x_test.reshape((full_x_test.shape[0], 28, 28, 1))

    elif constants.dataset_type in ["CIFAR10"]:
        (full_x_train, full_y_train), (full_x_test, full_y_test) = cifar10.load_data()
    elif constants.dataset_type in ["CIFAR100"]:
        (full_x_train, full_y_train), (full_x_test, full_y_test) = cifar100.load_data()
    elif constants.dataset_type in ["ImageNet32"]:
        # Note that this will work with Python3
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo)
            return dict

        def load_databatch(data_file, img_size=32):
            d = unpickle(data_file)
            x = d['data']
            y = d['labels']

            # Reshape input data
            full_x_test = []
            for sample in x:
                img_to_reshape_r = sample[:1024]
                img_to_reshape_r = img_to_reshape_r.reshape((32, 32, 1))
                img_to_reshape_g = sample[1024:(1024 * 2)]
                img_to_reshape_g = img_to_reshape_g.reshape((32, 32, 1))
                img_to_reshape_b = sample[(1024 * 2):]
                img_to_reshape_b = img_to_reshape_b.reshape((32, 32, 1))

                img = np.zeros((img_to_reshape_r.shape[0], img_to_reshape_r.shape[1], 3), np.uint8)
                img[:, :, [0]] = img_to_reshape_b
                img[:, :, [1]] = img_to_reshape_g
                img[:, :, [2]] = img_to_reshape_r

                full_x_test.append(img)

            full_x_test = np.asarray(full_x_test)
            # cv2.imwrite("test.png", full_x_test[2])

            # Labels are indexed from 1, shift it so that indexes start at 0
            y = [i - 1 for i in y]

            y = np.asarray(y)

            """
            mean_image = d['mean']

            x = x / np.float32(255)
            mean_image = mean_image / np.float32(255)
            
            # Remove average image from dataset
            x -= mean_image
            data_size = x.shape[0]

            img_size2 = img_size * img_size

            x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
            x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

            # Create mirrored images (Data augmentation)
            X_train = x[0:data_size, :, :, :]
            Y_train = y[0:data_size]
            X_train_flip = X_train[:, :, :, ::-1]
            Y_train_flip = Y_train
            X_train = np.concatenate((X_train, X_train_flip), axis=0)
            Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

            return dict(X_train=lasagne.utils.floatX(X_train), Y_train=Y_train.astype('int32'), mean=mean_image)
            """
            return full_x_test, y

        full_x_test, full_y_test = load_databatch(constants.validation_folder_path)

        full_x_train = []
        full_y_train = []
        for idx in range(1, 11):
            data_file = constants.training_folder_path + str(idx)
            full_x_train_tmp, full_y_train_tmp = load_databatch(data_file)
            full_x_train.append(full_x_train_tmp)
            full_y_train.append(full_y_train_tmp)

        full_x_train = np.concatenate(full_x_train, axis=0)
        full_y_train = np.concatenate(full_y_train, axis=0)

    full_x_train = full_x_train.astype('float32')

    # z-score
    mean = np.mean(full_x_train, axis=(0, 1, 2, 3))
    std = np.std(full_x_train, axis=(0, 1, 2, 3))
    full_x_train = (full_x_train - mean) / (std+1e-7)
    full_x_test = (full_x_test - mean) / (std+1e-7)

    twenty_train_data, twenty_train_labels = resample(full_x_train, full_y_train, n_samples=20000, replace=False, random_state=constants.seed_value)

    # twenty_train_labels_withoutKfold = np_utils.to_categorical(twenty_train_labels, constants.num_classes)

    history_solution = []
    # resultats = {}

    i = 0

    while i < constants.number_of_architecture:

        solution_archi = generate_new_architecture()
        if not(history_solution.__contains__(solution_archi)):
            history_solution.append(solution_archi)
            i += 1

    # print("**********")
    # i = len(history_solution)
    i = 0
    print(len(history_solution))

    if constants.top_N_path is not None:
        top_N = pickle.load(open(constants.top_N_path, "rb"))
    else:
        top_N = {
            "top_N_val_acc": [],
            "top_N_arch_id": [],
            "top_N_mean_std": [],
            "top_N_solution_arch": []
        }

    while i < len(history_solution):

        # f = open("./Results_CIFAR100_DGX2_V8Bis_20k_KFold.txt", "a")
        f = open(constants.results_output_file, "a")
        f.write(str(history_solution[i]) + "\n")
        f.write("-----------------------------------------------"+"\n")
        f.close()

        top_N, is_top_N_updated, top_acc, number_parameters = GM.trainWithKFOld(history_solution[i], twenty_train_data, twenty_train_labels, i, top_N)

        # f = open("./Results_ARCHI_CIFAR100_DGX2_V8Bis_20k_KFold.txt", "a")
        f = open(constants.architectures_output_file, "a")
        f.write(";".join(["TrainingImages_30000", "NetworkParameters_" + str(number_parameters), "KFold_" + str(constants.K_fold), str(top_acc), str(history_solution[i])]) + "\n")
        f.close()

        if is_top_N_updated:
            pickle.dump(top_N, open(constants.results_top_N, "wb"))
            utils_slack.send_top_N_to_slack(top_N)

        print("--------------------------------------------")

        i += 1

    # Create data folder in the next pipeline (2)
    os.makedirs("../../2_Architecture_Search_Kfold/data", exist_ok=True)
    os.makedirs("../../3_FDA/data", exist_ok=True)

    # Copy output results into the data folder from the next pipeline (2)

    # Source path
    # source = constants.results_top_N
    source = "/raid-dgx1/allanza/CodeOcean/FractalDecompositionAlgorithm/1_Architecture_Search/python/outputs/2022-09-15_13-41-43/Results_top_N.p"

    # Destination path
    destination = os.path.join("../../2_Architecture_Search_Kfold/data/", constants.results_top_N.split(os.sep)[-1])

    # Copy the content of
    # source to destination
    shutil.copyfile(source, destination)

    # Destination path
    destination = os.path.join("../../3_FDA/data/", constants.results_top_N.split(os.sep)[-1])

    # Copy the content of
    # source to destination
    shutil.copyfile(source, destination)

if __name__ == "__main__":
    main()
