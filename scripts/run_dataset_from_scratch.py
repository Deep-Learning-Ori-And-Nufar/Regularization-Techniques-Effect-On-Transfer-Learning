import subprocess
import sys
import os

from transfer_learning_utils import DATASETS, TRIMMING_SIZES, MAIN_SCRIPT_PATH, TRANSFER_LEARNING_EPOCH

def train_from_scratch(dataset_name, samples_per_class=0):
    print(f"\nStarting Training from scratch on {dataset_name}...")

    cmd = [
        sys.executable, MAIN_SCRIPT_PATH,
        "--train_type", "transfer",
        "--target_dataset_name", dataset_name,
        "--epochs", str(TRANSFER_LEARNING_EPOCH),
        "--should_train_from_scratch", str(1),
        "--samples_per_class", str(samples_per_class),
        "--full_train_set", "1"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished {dataset_name} completed successfully.")
    except subprocess.CalledProcessError:
        print(f"[Error] Failed to run {dataset_name}.")
    except KeyboardInterrupt:
        print("\n[Stopped] Interrupted by user.")
        sys.exit(1)

def main():
    if not os.path.exists(MAIN_SCRIPT_PATH):
        print(f"Error: Could not find {MAIN_SCRIPT_PATH}")
        return

    for dataset in DATASETS:
        for trimming_size in [TRIMMING_SIZES]:
            train_from_scratch(dataset, trimming_size)

    print("all experiments completed")

if __name__ == "__main__":
    main()