import subprocess
import sys

DATASETS = ["cifar10", "flowers102", "eurosat", "dtd"]
TRIMMING_SIZES = [10, 50, 100]
LOSSES = ["Cosine","VICReg","SIGReg", "CE"]
all_loss_configs = [
        {"loss": "CE",      "weight": None},
        {"loss": "Cosine",  "weight": 0.1},
        {"loss": "Cosine",  "weight": 1.0},
        {"loss": "VICReg",  "weight": 0.01},  
        {"loss": "VICReg",  "weight": 0.1},
        {"loss": "SIGReg",  "weight": 0.01},
    ]

MAIN_SCRIPT_PATH = "deep_learning_project_main_logic.py"
TRANSFER_LEARNING_EPOCH = 50

def run_experiment(loss_configs):
    for config in loss_configs:
        loss_name = config["loss"]
        reg_weight = config["weight"]
        samples_per_class = config.get("samples_per_class", 0)
        
        for dataset in DATASETS:
            print(f"\n==================================================")
            print(f"Running Experiment: {loss_name} with regularization_weight={reg_weight} for {dataset} dataset")
            print(f"==================================================")
            
            # Build the command arguments
            cmd = [
                sys.executable, MAIN_SCRIPT_PATH,
                "--loss_name", loss_name,
                "--target_dataset_name", dataset,
                "--train_type", "transfer",
                "--epochs", str(TRANSFER_LEARNING_EPOCH),
                "--should_train_from_scratch", "0",
                "--samples_per_class", str(samples_per_class),
                "--checkpoint_dir", "" if samples_per_class == 0 else "_trimmed_" + str(samples_per_class),
                "--full_train_set", "1"
            ]
            
            if reg_weight is not None:
                cmd.extend(["--reg_weight", str(reg_weight)])

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running {loss_name} on {dataset}.")
                print(f"Error details: {e}")

def run_trimmed_experiment(loss_configs, samples_per_class):
    for item in loss_configs:
        item["samples_per_class"] = samples_per_class
    
    run_experiment(loss_configs=loss_configs)