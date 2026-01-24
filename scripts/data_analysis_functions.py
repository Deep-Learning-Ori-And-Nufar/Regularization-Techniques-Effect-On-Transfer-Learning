import os
import torch
import pandas as pd
from transfer_learning_utils import DATASETS, all_loss_configs, TRIMMING_SIZES

from deep_learning_project_17_01 import (
    DATA_ROOT_DIR
)

def get_loss_name(loss):
    """Maps shorthand to the specific naming used in the training script."""
    mapping = {
        "CE": "Cross_Entropy",
        "Cosine": "CE_With_Cosine_Similarity",
        "VICReg": "CE_With_VICReg",
        "SIGReg": "CE_With_SIGReg",
        "Scratch18": "model_size_18",
        "Scratch50": "model_size_50"
    }
    return mapping.get(loss, loss)

def compare_val_results_at_epoch(loss_configs, checkpoint_dir, epoch=20):
    """
    Extracts validation accuracy from specific epoch checkpoints.
    Path structure: checkpoint_{LOSS}_epoch_{EPOCH}_reg_weight_{WEIGHT}.pth
    """
    all_results = []
    
    print(f"--- Extracting Validation Accuracy at Epoch {epoch} ---")

    for config in loss_configs:
        loss = config["loss"]
        weight = config["weight"]
        loss_name = get_loss_name(loss)
        
        # Construct the filename based on your description
        # Example: checkpoint_CE_With_Cosine_Similarity_epoch_20_reg_weight_10.0.pth
        filename = f"checkpoint_{loss_name}_epoch_{epoch}"
        
        if weight is not None:
            filename += f"_reg_weight_{weight}"
            
        # Assuming .pth extension based on typical usage
        filename += ".pth"
        
        full_path = os.path.join(checkpoint_dir, filename)

        if os.path.exists(full_path):
            try:
                # Load checkpoint on CPU
                checkpoint = torch.load(full_path, map_location='cpu')
                
                # Extract val_accs. Assuming it's a list, we take the last one (current epoch)
                val_accs = checkpoint.get('val_accs', [])
                
                if val_accs:
                    # Get the specific accuracy for this epoch (last in list)
                    current_val_acc = val_accs[-1]
                    
                    all_results.append({
                        "Loss Name": config["loss"],
                        "Regularization Weight": weight if weight is not None else "None",
                        "Validation Accuracy": round(current_val_acc, 2)
                    })
                else:
                    print(f"  [!] 'val_accs' list is empty in: {filename}")
                    
            except Exception as e:
                print(f"  [!] Error loading {filename}: {e}")
        else:
            print(f"  [!] Missing file: {filename}")

    if all_results:
        df_flat = pd.DataFrame(all_results)
        
        # Pivot table: Y=Weight, X=Loss Name
        df_pivot = df_flat.pivot(index="Regularization Weight", columns="Loss Name", values="Validation Accuracy")
        
        output_file = f"summary_val_acc_epoch_{epoch}.csv"
        df_pivot.to_csv(output_file)
        
        print(f"\nSuccessfully saved validation summary to {output_file}")
        print(df_pivot)
        return df_pivot
    else:
        print("No validation results found to compile.")
        return None


def compare_test_results(loss_configs, checkpoint_dir, is_transfer=True, trimming_size=0):
    """
    Generates a pivot-table CSV comparing test accuracy.
    X-axis: Loss + Regularization | Y-axis: Dataset name | Value: Accuracy
    """
    base_search_dir = os.path.join(checkpoint_dir, "transfer_results") if is_transfer else checkpoint_dir
    all_results = []
    
    # We always need to loop through configurations and datasets to avoid duplication
    for dataset in (DATASETS if is_transfer else ["ImageNet100"]):
        for config in loss_configs:
            loss = config["loss"]
            weight = config["weight"]
            loss_name = get_loss_name(loss)
            
            # Construct the path based on the logic in deep_learning_project_17_01.py
            if is_transfer:
                # Folder: dataset_cifar10_from_loss_Cross_Entropy_reg_weight_0.1
                if(loss.startswith("Scratch")):
                    exp_folder = f"dataset_{dataset}_from_scratch_{loss_name}"
                else:    
                    exp_folder = f"dataset_{dataset}_from_loss_{loss_name}"
                if(trimming_size != 0):
                    exp_folder += f"_trimming_to_{trimming_size}"
                if weight is not None:
                    exp_folder += f"_reg_weight_{weight}"
            
                # Filename: Cross_Entropytest_accuracy.pth (no underscore between name and 'test')
                test_acc_path = os.path.join(base_search_dir, exp_folder, "Cross_Entropy_test_accuracy.pth")
            else:
                # Filename: CE_With_VICReg_regularization_weight_1.0test_accuracy.pth
                filename = f"{loss_name}"
                if weight is not None:
                    filename += f"_regularization_weight_{weight}"
                filename += "_test_accuracy.pth"
                test_acc_path = os.path.join(base_search_dir, filename)

            # Load accuracy if the file exists
            if os.path.exists(test_acc_path):
                data = torch.load(test_acc_path, map_location='cpu')
                test_acc = data.get('test_accuracy', "N/A")
                
                # Create a clean X-axis label
                config_label = f"{loss} Reg weight: {weight}" if weight is not None else loss
                
                all_results.append({
                    "Dataset": dataset,
                    "Config": config_label,
                    "Accuracy": round(test_acc,2),
                    "Regularization Weight":weight,
                    "Loss Name":config["loss"]

                })
            else:
                print(f"  [!] Missing file: {test_acc_path}")

    if all_results:
        df_flat = pd.DataFrame(all_results)
        
        if(is_transfer):
            # Reshape to meet your requirement: Y=Dataset, X=Config
            df_pivot = df_flat.pivot(index="Dataset", columns="Config", values="Accuracy")
        else:
            df_pivot = df_flat.pivot(index="Regularization Weight", columns="Loss Name", values="Accuracy")

        
        suffix = "transfer" if is_transfer else "full_train"
        suffix += f"_trimming_{trimming_size}" if (trimming_size !=0) else ""
        output_file = f"summary_table_{suffix}.csv"
        df_pivot.to_csv(output_file)
        
        print(f"\nSuccessfully saved summary table to {output_file}")
        return df_pivot
    else:
        print("No results found to compile.")
        return None
    
if __name__ == "__main__":
    #for saving trimming results
    # final_configs = all_loss_configs + [
    #     {"loss": "Scratch18", "weight": None},
    #     {"loss": "Scratch50", "weight": None}
    # ]
    # for trimming_size in TRIMMING_SIZES:
    #     compare_test_results(loss_configs=final_configs, checkpoint_dir=os.path.join(DATA_ROOT_DIR,"checkpoints"), is_transfer=True, trimming_size=trimming_size)
    # our_loss_configs = [
    #     {"loss": "CE",      "weight": None},
    #     {"loss": "Cosine",  "weight": 0.01},
    #     {"loss": "Cosine",  "weight": 0.1},
    #     {"loss": "Cosine",  "weight": 1.0},
    #     {"loss": "Cosine",  "weight": 10.0},
    #     {"loss": "VICReg",  "weight": 0.01},
    #     {"loss": "VICReg",  "weight": 0.1},
    #     {"loss": "VICReg",  "weight": 1.0},
    #     {"loss": "VICReg",  "weight": 10.0},
    #     {"loss": "SIGReg",  "weight": 0.01},
    #     {"loss": "SIGReg",  "weight": 0.1},
    #     {"loss": "SIGReg",  "weight": 1.0},
    #     {"loss": "SIGReg",  "weight": 10.0},
    # ]

    # checkpoints_path = os.path.join(DATA_ROOT_DIR, "checkpoints")
    
    # compare_val_results_at_epoch(
    #     loss_configs=our_loss_configs, 
    #     checkpoint_dir=checkpoints_path, 
    #     epoch=20
    # )
    #to check the hyper parameter tuning
    # compare_test_results(loss_configs=our_loss_configs, checkpoint_dir=os.path.join(DATA_ROOT_DIR,"checkpoints"), is_transfer=False)
    # all_loss_configs.append({"loss": "Scratch18",  "weight": None})
    # all_loss_configs.append({"loss": "Scratch50",  "weight": None})
    # compare_test_results(loss_configs=all_loss_configs, checkpoint_dir=os.path.join(DATA_ROOT_DIR,"checkpoints"), is_transfer=True)
    compare_test_results(loss_configs=all_loss_configs, checkpoint_dir=os.path.join(DATA_ROOT_DIR,"checkpoints"), is_transfer=False)
