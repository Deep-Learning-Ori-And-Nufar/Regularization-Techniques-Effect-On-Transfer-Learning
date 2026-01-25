import subprocess
from transfer_learning_utils import LOSSES, MAIN_SCRIPT_PATH


epochs = 20
reg_weights = [0.01, 0.1, 1, 10]
for loss in LOSSES:
    if loss == "CE":
        current_weights = [None] 
    else:
        current_weights = reg_weights
    for weight in current_weights:
        
        print(f"\n" + "="*60)
        print(f"STARTING TRAINING | Loss: {loss} | Reg Weight: {weight}")
        print("="*60)
        
        command = [
            "python", MAIN_SCRIPT_PATH,
            f"--loss_name={loss}",
            f"--epochs={epochs}",
            "--is_hyperparam_tuning=1",
            "--train_type=full_train"
        ]

        if weight is not None:
            command.extend(["--reg_weight", str(weight)])

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during training for weight {weight}: {e}")
            continue

    print("\nAutomation complete. All specified reg_weights have been processed.")