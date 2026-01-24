<h1 align="center">
  <br>
  Evaluating Regularization Methods Effect on Transfer Learning Performance
  <br>
</h1>

<p align="center">
  <strong>Nufar Cohen</strong> •
  <strong>Ori Fridman</strong>
</p>

<h4 align="center">Official repository for evaluating the impact of SSL-inspired regularization techniques on transfer learning performance</h4>

---

## Table of Contents
1. [Abstract](#abstract)
2. [Prerequisites](#prerequisites)
3. [Repository Organization](#repository-organization)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Features](#features)
7. [Experiments](#experiments)
8. [Results & Key Findings](#results--key-findings)
9. [Conclusions](#conclusions)


---
## Abstract

This project evaluates how different regularization techniques applied to CNN features during training affect transfer learning accuracy on downstream tasks. We compare standard Cross Entropy (CE) with CE combined with Cosine Similarity, VICReg, and SIGReg regularization, training on ImageNet100 and transferring to CIFAR10, Flowers102, EuroSAT, and DTD.

## Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for training

## Repository Organization

| File/Directory | Content |
|---------------|---------|
| `deep_learning_project_17_01.py` | Main script for full training on ImageNet100 and transfer learning experiments |
| `data_analysis_functions.py` | Generates CSV summary tables from experiment results |
| `image_visualization.py` | Visualizes data augmentations and training metrics |
| `scripts/` | Directory containing automation scripts |
| `scripts/run_hyper_parameters.py` | Automates hyperparameter tuning across regularization weights |
| `scripts/run_transfer_trimmed.py` | Runs transfer learning with dataset trimming |
| `scripts/run_dataset_from_scratch.py` | Trains models from scratch on downstream datasets |
| `scripts/transfer_learning_utils.py` | Utilities for transfer learning experiments |
| `scripts/plot_functions.py` | Plots training results and confusion matrices |
| `csv_results/` | Directory for CSV summary tables |
| `plots/` | Directory for generated plots and visualizations |
| `Project_Report_24_01.docx` | Detailed project report |
| `summary_val_acc_epoch_20.csv` | Validation accuracy summaries |

## Installation

### Dependencies
```bash
pip install torch torchvision
pip install numpy matplotlib seaborn scikit-learn
pip install kornia pandas pillow
```

### Data Setup
1. Download ImageNet100 from kaggle and organize as:
```
DATA_ROOT_DIR/
├── train.X1/ (class folders)
├── train.X2/
├── train.X3/
├── train.X4/
├── val.X/
└── Labels.json
```

2. Other datasets (CIFAR10, Flowers102, EuroSAT, DTD) downloadedS automatically via torchvision.

## Usage

### Full Training on ImageNet100
Train ResNet50 from scratch with regularization:
```bash
python deep_learning_project_17_01.py --loss_name VICReg --reg_weight 0.1 --train_type full_train --epochs 60
```

Available loss options: `CE`, `Cosine`, `VICReg`, `SIGReg`

### Hyperparameter Tuning
Run automated hyperparameter search:
```bash
python scripts/run_hyper_parameters.py
```

### Transfer Learning
Transfer pre-trained weights to downstream datasets:
```bash
python deep_learning_project_17_01.py --loss_name Cosine --reg_weight 0.1 --target_dataset_name cifar10 --train_type transfer --epochs 20
```

### Transfer with Dataset Trimming
Limit samples per class for small dataset simulation:
```bash
python deep_learning_project_17_01.py --loss_name VICReg --reg_weight 1.0 --target_dataset_name flowers102 --train_type transfer --samples_per_class 50 --epochs 20
```

### Train from Scratch (No Pre-training)
Baseline comparison without transfer learning:
```bash
python deep_learning_project_17_01.py --target_dataset_name eurosat --train_type transfer --should_train_from_scratch 1 --epochs 20
```

### Automated Transfer Experiments
Run all transfer learning configurations:
```bash
python scripts/run_transfer_trimmed.py --samples_per_class 10
```

Train from scratch on all datasets:
```bash
python scripts/run_dataset_from_scratch.py
```

### Analysis and Visualization
Generate result summaries:
```bash
python data_analysis_functions.py
```

Visualize augmentations:
```bash
python image_visualization.py
```

Plot training curves:
```bash
python scripts/plot_functions.py
```

## Features

- Custom ResNet implementation (18/50 layers)
- SSL-inspired regularization: Cosine Similarity, VICReg, SIGReg
- Automated transfer learning pipeline
- Kornia-based data augmentation
- Checkpoint management with resume capability
- Comprehensive evaluation with confusion matrices

## Experiments

### Regularization Comparison
- **Cosine Similarity**: Penalizes feature correlations
- **VICReg**: Variance-Invariance-Covariance regularization
- **SIGReg**: Isotropic Gaussian regularization

### Transfer Learning Evaluation
- Frozen backbone fine-tuning
- Performance across domains: objects, textures, satellite, flowers
- Dataset trimming studies (10, 50, 100 samples/class)

## Results & Key Findings
- Most Regularizations improved ImageNet100 generalization
- VICReg achieved best results for transfer learning

<b>Transfer Learning Test Accuracy Without Trimming: </b>

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th rowspan="2">CE (Baseline)</th>
      <th colspan="2">Cosine Similarity</th>
      <th>SIGReg</th>
      <th colspan="2">VICReg</th>
      <th colspan="2">From Scratch</th>
    </tr>
    <tr>
      <th>λ=0.1</th>
      <th>λ=1.0</th>
      <th>λ=0.01</th>
      <th>λ=0.01</th>
      <th>λ=0.1</th>
      <th>ResNet18</th>
      <th>ResNet50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Cifar10</b></td>
      <td>79.94</td>
      <td>78.16</td>
      <td>71.46</td>
      <td>76.31</td>
      <td>81.09</td>
      <td>78.1</td>
      <td><b>85.23</b></td>
      <td>83.81</td>
    </tr>
    <tr>
      <td><b>DTD</b></td>
      <td>39.68</td>
      <td>37.87</td>
      <td>33.78</td>
      <td>36.06</td>
      <td><b>39.41</b></td>
      <td>35.21</td>
      <td>23.46</td>
      <td>17.87</td>
    </tr>
    <tr>
      <td><b>EuroSat</b></td>
      <td>92.07</td>
      <td>90.76</td>
      <td>82.81</td>
      <td>89.85</td>
      <td>94.15</td>
      <td>91.48</td>
      <td><b>98.59</b></td>
      <td>98.19</td>
    </tr>
    <tr>
      <td><b>Flowers102</b></td>
      <td>58.82</td>
      <td>56.71</td>
      <td>40.66</td>
      <td>46.56</td>
      <td><b>59.96</b></td>
      <td>53.07</td>
      <td>34.98</td>
      <td>29.22</td>
    </tr>
  </tbody>
</table>

<b>Test Accuracy Trimming Datasets to 10 samples per class:</b>
<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th rowspan="2">CE (Baseline)</th>
      <th colspan="2">Cosine Similarity</th>
      <th>SIGReg</th>
      <th colspan="2">VICReg</th>
      <th colspan="2">From Scratch</th>
    </tr>
    <tr>
      <th>λ=0.1</th>
      <th>λ=1.0</th>
      <th>λ=0.01</th>
      <th>λ=0.01</th>
      <th>λ=0.1</th>
      <th>ResNet18</th>
      <th>ResNet50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Cifar10</b></td>
      <td>49.73</td>
      <td>44.85</td>
      <td>39.44</td>
      <td>39.62</td>
      <td><b>49.95</b></td>
      <td>45.03</td>
      <td>25.57</td>
      <td>20.01</td>
    </tr>
    <tr>
      <td><b>DTD</b></td>
      <td>30.85</td>
      <td>29.84</td>
      <td>26.01</td>
      <td>28.14</td>
      <td><b>32.07</b></td>
      <td>26.6</td>
      <td>13.62</td>
      <td>10.74</td>
    </tr>
    <tr>
      <td><b>EuroSat</b></td>
      <td>71.56</td>
      <td>68.98</td>
      <td>59.15</td>
      <td>64.33</td>
      <td><b>73.44</b></td>
      <td>71.13</td>
      <td>53.06</td>
      <td>39.06</td>
    </tr>
  </tbody>
</table>

## Configuration

- **Batch Size**: 128 (training), 32 (transfer)
- **Learning Rate**: 0.01 with cosine annealing
- **Weight Decay**: 1e-4
- **Input Size**: 224x224
- **Augmentation**: Kornia pipeline

## Conclusions

Regularization Impact: Adding SSL-inspired constraints to latent features positively influences transfer learning.

VICReg Superiority: VICReg was the only method to consistently surpass the Cross-Entropy baseline across various datasets.

Inverse Correlation: An inverse correlation was observed between pretraining accuracy and transfer learning effectiveness; methods like SIGReg and Cosine Similarity improved source accuracy but failed to enhance transfer performance.

Feature Quality: Regularization helps the model learn fine-grained details and "feature libraries" that are useful for domain shifts and small-scale datasets.
