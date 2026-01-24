
from transfer_learning_utils import run_trimmed_experiment, all_loss_configs
import argparse

if __name__ == "__main__":
     # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_class", default=10, type=int, help='get the amount of samples to train with')
    args = parser.parse_args()
    run_trimmed_experiment(all_loss_configs, samples_per_class=args.samples_per_class)