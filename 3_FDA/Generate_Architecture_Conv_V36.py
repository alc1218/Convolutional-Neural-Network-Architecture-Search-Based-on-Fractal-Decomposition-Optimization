import run_V2 as run
#import keras
from keras.datasets import cifar10, cifar100, mnist
import numpy as np
from keras.utils import np_utils
import os
import random
import tensorflow as tf

import generateModel as GM
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import pickle
import constants


def generateNewArchitecture():
    number_of_layers = [1,2,3,4]
    number_of_conv = [2,3,4]


    nb_kernel = [5,6,7,8,9]
    kernel_size = [2,3,4,5]
    stride = [1]

    maxPooling_size = [2]


    number_of_layers_choice = random.choice(number_of_layers)
    final_result = [None]  * number_of_layers_choice



    print(number_of_layers_choice)
    print("***")

    for i in range(number_of_layers_choice):
        #print("--"+str(i))
        number_of_conv_choice = random.choice(number_of_conv)
        #print("----"+str(number_of_conv_choice))
        final_result[i] = []


        #print("--**")

        for j in range(number_of_conv_choice):

            nb_kernel_choice = random.choice(nb_kernel)
            kernel_size_choice = random.choice(kernel_size)
            stride_choice = random.choice(stride)

            final_result[i].append([2**nb_kernel_choice,kernel_size_choice,stride_choice])
            #print("----"+str([2**nb_kernel_choice,kernel_size_choice,stride_choice]))
        maxPooling_size_choice = random.choice(maxPooling_size)
        final_result[i].append(maxPooling_size_choice)
        #print("----"+str([maxPooling_size_choice,maxPooling_size_choice]))

    #print("***")
    #print(final_result)
    #print("****************************************************")
    return final_result



num_classes = constants.datasets[constants.dataset_type]["num_classes"]

# READ DATA FRAME
if constants.dataset_type in ["MNIST"]:
    (full_x_train, full_y_train), (full_x_test, full_y_test) = mnist.load_data()
    full_x_train = full_x_train.reshape((full_x_train.shape[0], 28, 28, 1))
    full_x_test = full_x_test.reshape((full_x_test.shape[0], 28, 28, 1))

elif constants.dataset_type in ["CIFAR10"]:
    (full_x_train, full_y_train), (full_x_test, full_y_test) = cifar10.load_data()
elif constants.dataset_type in ["CIFAR100"]:
    (full_x_train, full_y_train), (full_x_test, full_y_test) = cifar100.load_data()

full_x_train = full_x_train.astype('float32')

#z-score
mean = np.mean(full_x_train,axis=(0,1,2,3))
std = np.std(full_x_train,axis=(0,1,2,3))
full_x_train = (full_x_train-mean)/(std+1e-7)
full_x_test = (full_x_test-mean)/(std+1e-7)




full_x_train, full_x_test, full_y_train, full_y_test = train_test_split(full_x_train, full_y_train, test_size=0.2)

full_y_train = np_utils.to_categorical(full_y_train, num_classes)
full_y_test = np_utils.to_categorical(full_y_test, num_classes)



history_solution = []
resultats = {}

i = 0

# Load top_N
if constants.top_N_path is not None:
    top_N = pickle.load(open(constants.top_N_path, "rb"))
else:
    raise NameError("Top_N structure does not exist: " + str(constants.top_N_path))

final_scores = [sample[-1] for sample in top_N["top_N_val_acc"]]
architectures = [sample for sample in top_N["top_N_solution_arch"]]

zipped_lists = zip(final_scores, architectures)
sorted_pairs = sorted(zipped_lists, reverse=True)

tuples = zip(*sorted_pairs)
final_scores, architectures = [ list(tuple) for tuple in  tuples]

while i < constants.numberOfArchitecture:

    # solution_archi = [[[256, 3, 1], [256, 2, 1], 2], [[256, 4, 1], [256, 3, 1], [256, 5, 1], 2], [[256, 5, 1], [256, 5, 1], 2], [[256, 2, 1], [256, 2, 1], [256, 2, 1], 2]]

    # TF1 CIFAR100
    # solution_archi = [[[256, 3, 1], [256, 4, 1], [256, 2, 1], 2], [[256, 2, 1], [256, 5, 1], [256, 4, 1], 2], [[256, 4, 1], [256, 3, 1], 2]]  # 0.5711666666666666
    # solution_archi = [[[256, 2, 1], [256, 3, 1], [256, 3, 1], [256, 2, 1], 2], [[256, 3, 1], [256, 5, 1], 2], [[256, 2, 1], [256, 3, 1], [256, 3, 1], 2]]  # 0.5710000000000001
    # solution_archi = [[[256, 2, 1], [256, 3, 1], 2], [[256, 2, 1], [256, 3, 1], [256, 4, 1], 2], [[256, 3, 1], [256, 4, 1], 2]]  # 0.5643333333333332

    # TF2 CIFAR100
    # solution_archi = [[[256, 3, 1], [256, 2, 1], [256, 3, 1], 2], [[256, 5, 1], [256, 3, 1], 2], [[256, 3, 1], [256, 4, 1], 2]]  # 0.590666671593984
    # solution_archi = [[[256, 2, 1], [256, 2, 1], 2], [[256, 2, 1], [256, 4, 1], [256, 5, 1], [256, 2, 1], 2], [[256, 4, 1], [256, 4, 1], 2]]  # 0.5866666833559672
    # solution_archi = [[[256, 3, 1], [256, 3, 1], 2], [[256, 2, 1], [256, 3, 1], 2], [[256, 2, 1], [256, 2, 1], [256, 5, 1], 2]]  # 0.5848333239555359

    # constants.architectures[constants.architecture_type]["solution_archi"]
    solution_archi = architectures[i]

    if not(history_solution.__contains__(solution_archi)):
        # f = open("./Results_MNIST_2D.txt", "a")
        f = open(constants.output_file, "a")
        f.write(str(solution_archi)+"\n")
        f.write("-----------------------------------------------"+"\n")
        f.close()
        
        history_solution.append(solution_archi)
        
        print(solution_archi)
        dim = 7 + len(solution_archi)
        for boucle in range(len(solution_archi)):
            dim = dim + (len(solution_archi[boucle])-1)
        
        print(dim)
        #print("Dimension : 3 + " +str(len(solution_archi))+" soit :"+str(dim))
        print("LAUNCH WITH ARCHITECTURE : " + str(solution_archi))
        print(len(full_x_train))
        print(len(full_y_train))
        print(len(full_x_test))
        print(len(full_y_test))
       
    
        res = run.run(solution_archi,dim,full_x_train,full_y_train,full_x_test,full_y_test)
        resultats[str(solution_archi)] = tuple(res)
        print("RESULTS OF ARCHITECTURE : " + str(res))
        i = i+1
        print("--------------------------------------------")
        
#model = NGM.get_model_withParam(archi,[2,2])
#model.summary()


#for i in range(decoder_nb_layer):



# Source path
# source = constants.results_top_N
source = constants.output_file

# Destination path
destination = os.path.join("../../results/", constants.output_file.split(os.sep)[-1])

# Copy the content of
# source to destination
shutil.copyfile(source, destination)
