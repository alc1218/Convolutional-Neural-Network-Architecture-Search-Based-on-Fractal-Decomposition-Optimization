import keras
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
import random
import tensorflow as tf
import os
import math
from sklearn.model_selection import StratifiedKFold
from copy import copy
from kerascosineannealing.cosine_annealing import CosineAnnealingScheduler

from utils_gpu import pick_gpu_lowest_memory
import constants

os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())


def get_model_withParam(solution_archi,solution_optim,inputShape):
    print("ENTER NEW GENERATE MODEL")
    decoder_nb_layer = len(solution_archi)
    #print(decoder_nb_layer)
    #print("***")
    #for i in range(decoder_nb_layer):
    #    number_of_convlayer = len(solution_archi[i])-1
    #    print("--"+str(i))
    #    print("----"+str(number_of_convlayer))
    #    print("--**")
    #    for j in range(number_of_convlayer):
    #        print("----"+str(solution_archi[i][j]))
    #    print("----"+str(solution_archi[i][number_of_convlayer]))
        
    
    #weight_decay = 1e-4
    
    wd = solution_optim[1]
    weight_decay = wd*0.1
    
    num_classes = constants.datasets[constants.dataset_type]["num_classes"]
    
    model = Sequential()
    
    first = False
    
    decoder_nb_layer = len(solution_archi)
    
    boucle_kernel_number = 0+7+decoder_nb_layer
    
    for i in range(decoder_nb_layer):
        number_of_convlayer = len(solution_archi[i])-1
        for j in range(number_of_convlayer):
            temp_nb_kernel = solution_optim[boucle_kernel_number]
            boucle_kernel_number = boucle_kernel_number+1
            nb_kernel = int(temp_nb_kernel * math.ceil(480*temp_nb_kernel + 32))
            kernel_size = solution_archi[i][j][1]
            stride = solution_archi[i][j][2]
        
            if(first == False):
                model.add(Conv2D(nb_kernel, (kernel_size,kernel_size), strides=(stride,stride), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=inputShape))
                first = True
            else:
                model.add(Conv2D(nb_kernel, (kernel_size,kernel_size), strides=(stride,stride), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
        
        maxPoolSize = solution_archi[i][number_of_convlayer]
        model.add(MaxPooling2D(pool_size=(maxPoolSize,maxPoolSize)))
        model.add(Dropout(solution_optim[i+7]))
       
    model.add(Flatten())
    model.add(Dropout(solution_optim[5]))
    model.add(BatchNormalization())
    temp_dense = solution_optim[6]
    nb_dense = int(temp_dense * math.ceil(3990*temp_dense + 10))
    model.add(Dense(nb_dense))    
    model.add(Dense(num_classes, activation='softmax'))
    
    t = solution_optim[0]
    new_lr = (10**-3/8)*((81**t)-1)
    
    sgd = keras.optimizers.SGD(lr=new_lr, momentum=solution_optim[2],nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model


if constants.dataset_type in ["CIFAR10", "CIFAR100"]:
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
elif constants.dataset_type in ["MNIST"]:
    datagen_1 = ImageDataGenerator(
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=5,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=5,
            channel_shift_range=0.1,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=False,  # randomly flip images
            validation_split=0.0)

datagen_2 = copy(datagen_1)


def unison_shuffling_copies(a,b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    
    return a[p], b[p]

def mixing_generator_2(generator_1):
    while True:
        (batch_x_1, batch_y_1) = next(generator_1)
        (batch_x_2, batch_y_2) = unison_shuffling_copies(batch_x_1, batch_y_1)
        
        #alpha = 1
        alpha = 0.4
        
        lam = np.random.beta(alpha, alpha)
        batch_x = (lam * batch_x_1 + (1. - lam) * batch_x_2)
        batch_y = (lam * batch_y_1 + (1. - lam) * batch_y_2)
        
        yield batch_x,batch_y

def trainWithoutKFOld(solution_archi,solution_optim,train_data,train_labels, test_data, test_labes):
    num_classes = constants.datasets[constants.dataset_type]["num_classes"]
    epochs = 30
    K_fold = 5
    warmup_epoch = 5
    
    seed_value= 10
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    T = math.ceil(399*solution_optim[3] + 1)
    
    t = solution_optim[0]
    new_lr = (10**-3/8)*((81**t)-1)
       
    x = solution_optim[4]
    batch_size = math.ceil(480*x + 32)
    steps = math.ceil(len(train_data)/batch_size)
    
    
    reduce_lr = CosineAnnealingScheduler(T_max=T, eta_max=new_lr, eta_min=1e-5)
    callbacks_list = [reduce_lr]
    
    datagen_1.fit(train_data)
    datagen_1_flow = datagen_1.flow(train_data, train_labels, batch_size=batch_size)
    datagen_flow = mixing_generator_2(datagen_1_flow)
    
    model = get_model_withParam(solution_archi,solution_optim,train_data.shape[1:])
    
    history = model.fit_generator(datagen_1_flow,epochs=warmup_epoch,validation_data= (test_data,test_labes),steps_per_epoch=steps)

    
    history = model.fit_generator(datagen_flow,epochs=epochs-warmup_epoch,validation_data=(test_data,test_labes),steps_per_epoch=steps,callbacks=callbacks_list)

    
    nb_epochs = len(history.history["val_acc"])
    result = history.history["val_acc"][nb_epochs-1]

    #print(np.mean(all_scores))
    K.clear_session()
    print("EXIT trainWithKFOld")
    return result