import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_hyperparameter_results(csv_path, ce_val_acc=None):
    """
    Reads a CSV containing regularization results and plots 
    Validation Accuracy vs Regularization Weight (Log Scale).
    
    Args:
        csv_path (str): Path to the summary CSV.
        ce_val_acc (float): The baseline Validation Accuracy for standard Cross-Entropy.
    """

    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' was not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df_melted = df.melt(
        id_vars='Regularization Weight', 
        var_name='Technique', 
        value_name='Validation Accuracy'
    )


    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Scatter plot for the points
    sns.scatterplot(
        data=df_melted, 
        x='Regularization Weight', 
        y='Validation Accuracy', 
        hue='Technique', 
        style='Technique', 
        s=100,
        alpha=0.7, 
        palette='deep'
    )

    # Add Horizontal Baseline line for CE
    if ce_val_acc is not None:
        plt.axhline(y=ce_val_acc, color='r', linestyle='-', linewidth=1, label=f'Baseline CE ({ce_val_acc}%)')
        plt.legend(title='Regularization Technique', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.text(df_melted['Regularization Weight'].min(), ce_val_acc + 0.5, 
             'Baseline Cross Entropy', color='r', fontweight='bold', va='bottom')

    plt.xscale('log')
    plt.ylim(0, 70)
    plt.title('Validation Accuracy vs. Regularization Weight', fontsize=16)
    plt.xlabel('Regularization Weight (Log Scale)', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_hyperparameter_results_wrapper():
    csv_filename = f"csv_results/summary_table_full_train_hyper_parameter.csv"
    
    print(f"Generating plot from: {csv_filename}...")
    #this is not saved in the csv, so we hardcode it here
    CE_val_acc = 62
    plot_hyperparameter_results(csv_filename, CE_val_acc)

def build_transfer_learning_plots(paths_array):
    """
    Args:
        paths_array: List of strings [path_to_10_samples, path_to_50_samples, 
                                     path_to_100_samples, path_to_full_data]
    """
    x_axis_placement = [0, 50, 100, 150] 
    labels = ['10', '50', '100', 'Full']
    
    # Load all CSVs into dataframes
    dfs = [pd.read_csv(p) for p in paths_array]
    
    datasets = sorted(set().union(*[df['Dataset'].unique() for df in dfs]))
    
    techniques = [
        'VICReg Reg weight: 0.1', 
        'VICReg Reg weight: 0.01', 
        'SIGReg Reg weight: 0.01', 
        'Cosine Reg weight: 0.1', 
        'Cosine Reg weight: 1.0', 
        'CE', 
        'Scratch18',  
        'Scratch50'  
    ]
    color_map = {
        'CE': '#333333',       # Dark Gray/Black
        'COSINE': '#1f77b4',   # Blue
        'SIGREG': '#ff7f0e',   # Orange
        'VICREG': '#2ca02c',   # Green
        'SCRATCH': '#d62728'   # Red
    }
    for ds in datasets:
        if sum(1 for df in dfs if ds in df['Dataset'].values) < 2:
            continue
            
        plt.figure(figsize=(10, 6))
        
        for tech in techniques:
            accuracies = []
            valid_x = []
            
            for i, df in enumerate(dfs):
                if ds in df['Dataset'].values and tech in df.columns:
                    acc = df.loc[df['Dataset'] == ds, tech].values[0]
                    accuracies.append(acc)
                    valid_x.append(x_axis_placement[i])
            
            if accuracies:
                tech_upper = tech.upper()
                line_color = '#7f7f7f' # Default Gray
                
                # Assign color based on prefix check
                for prefix, color in color_map.items():
                    if tech_upper.startswith(prefix):
                        line_color = color
                        break
                
                display_label = tech
                if tech == 'Scratch18':
                    display_label = 'Scratch ResNet18'
                elif tech == 'Scratch50':
                    display_label = 'Scratch ResNet50'

                # Style logic
                is_scratch = 'SCRATCH' in tech_upper
                style = '--' if is_scratch else '-'
                if tech_upper == 'CE':
                    mkr = 'D'  # Diamond for the baseline CE
                elif 'SCRATCH' in tech_upper:
                    mkr = 's'  # Square for models trained from scratch
                else:
                    mkr = 'o'  # Circle for all other regularization techniques (VICReg, SIGReg, Cosine)
                
                alpha_val = 0.4 if ('VICReg Reg weight: 0.1' in tech or 'Cosine Reg weight: 1.0' in tech or '50' in tech) else 1.0
                plt.plot(valid_x, accuracies, label=display_label, color=line_color,
                         marker=mkr, linestyle=style, linewidth=2, alpha=alpha_val)
                       
        plt.title(f'Transfer Learning Efficiency: {ds.upper()}', fontsize=14)
        plt.xlabel('Training Samples Per Class', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.xticks(x_axis_placement, labels)
        plt.gca().invert_xaxis()
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(
            loc='lower left', 
            frameon=True, 
            framealpha=0.8, 
            ncol=1 
        )
        plt.tight_layout()
        plt.show()

def build_transfer_learning_plots_wrapper():
    paths = [
        "csv_results/summary_table_transfer_trimming_10.csv",
        "csv_results/summary_table_transfer_trimming_50.csv",
        "csv_results/summary_table_transfer_trimming_100.csv",
        "csv_results/summary_table_transfer.csv"
    ]
    build_transfer_learning_plots(paths)

def main():
    build_transfer_learning_plots_wrapper()
    # plot_hyperparameter_results()

if __name__ == "__main__":
    main()