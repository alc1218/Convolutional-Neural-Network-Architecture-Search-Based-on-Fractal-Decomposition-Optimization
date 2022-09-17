import os
import time
import datetime
import tensorflow as tf
import numpy as np
from math import log, exp


root_output_folder = "outputs"
os.makedirs(root_output_folder, exist_ok=True)

# Create timestamp
timestamp = time.time()
readable_datetime = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(os.path.join(root_output_folder, readable_datetime), exist_ok=True)

# Select proper dataset
dataset_type = "CIFAR10" # "MNIST", "CIFAR10", "CIFAR100"

results_output_file = os.path.join(root_output_folder, readable_datetime, "Results_" + str(dataset_type) + "_V8Bis_20k_KFold.txt")
architectures_output_file = os.path.join(root_output_folder, readable_datetime, "Results_ARCHI_" + str(dataset_type) + "_V8Bis_20k_KFold.txt")
results_top_N = os.path.join(root_output_folder, readable_datetime, "Results_top_N.p")

# CSV logger callback
csv_logger_folder = "csv_logger"
csv_logger_output_folder = os.path.join(root_output_folder, readable_datetime, csv_logger_folder)
os.makedirs(csv_logger_output_folder, exist_ok=True)

# Early stopping callback
top_N_path = None
if dataset_type in ["CIFAR10"]:
    num_classes = 10

elif dataset_type in ["CIFAR100"]:
    # top_N_path = "/raid-dgx1/allanza/FDA/FractalDecompositionAlgorithm/Architecture_Search_Gecco/outputs/2020-05-01_13-05-34/Results_top_N.p"
    # top_N_path = "/raid-dgx1/allanza/FDA/FractalDecompositionAlgorithm/Architecture_Search_Gecco/outputs/2020-05-01_13-05-34/Results_top_N_50_0.1.p"
    top_N_path = "/raid-dgx1/nshvai/FDA/FractalDecompositionAlgorithm/Architecture_Search_Gecco/outputs/Results_top_N_start_20200518.p"
    num_classes = 100

elif dataset_type in ["MNIST"]:
    num_classes = 10

elif dataset_type in ["ImageNet32"]:
    training_folder_path = "/raid-dgx1/allanza/FDA/FractalDecompositionAlgorithm/1_Architecture_Search/data/DB/ImageNet32/train_data_batch_"
    validation_folder_path = "/raid-dgx1/allanza/FDA/FractalDecompositionAlgorithm/1_Architecture_Search/data/DB/ImageNet32/val_data"
    top_N_path = None
    num_classes = 1000

# Random Search hyperparameters
number_of_architecture = 100
top_N = 50
max_epoch = 30
K_fold = 10  #
batch_size = 64
max_nb_runs = 1  # 2

# percentile_threshold = 0.1  ## previous version with single percentile threshold
perc_min = 0.02
perc_max = 0.5
percentile_threshold = [exp(x) for x in np.linspace(log(perc_min), log(perc_max), num=max_epoch)]

# Model hyperparameters
weight_decay = 1e-4
new_lr = 0.01

# Script hyperparameters
seed_value = 42
val_acc_name = "val_acc" if tf.__version__[0] == "1" else "val_accuracy"

# Slack
# slack_url = None
slack_url = "https://hooks.slack.com/services/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"
