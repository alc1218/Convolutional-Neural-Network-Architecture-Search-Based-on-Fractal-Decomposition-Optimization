import os
from datetime import datetime

dataset_type = "CIFAR10"  # "MNIST", "CIFAR10", "CIFAR100"
datasets = {
    "MNIST": {
        "num_classes": 10
    },
    "CIFAR10": {
        "num_classes": 10
    },
    "CIFAR100": {
        "num_classes": 100
    },
}

top_N_path = "data/Results_top_N.p"
numberOfArchitecture = 1

root_output_folder = "./outputs"
os.makedirs(root_output_folder, exist_ok=True)
os.makedirs(os.path.join(root_output_folder, dataset_type), exist_ok=True)

datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join(root_output_folder, dataset_type, datetime_str), exist_ok=True)

output_file = os.path.join(root_output_folder, dataset_type, datetime_str, "final_results.txt")
