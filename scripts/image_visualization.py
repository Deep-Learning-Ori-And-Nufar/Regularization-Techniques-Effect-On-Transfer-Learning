import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
from deep_learning_project_main_logic import (
    ImageNet100Dataset, 
    get_augmentation_pipeline, 
    DATA_ROOT_DIR, 
    get_imagenet_dataloaders
)

# Standard ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def show_dataset_comparison(dataset_obj, loader, augmentation_pipeline, num_images=10, save_path="figures"):
    """
    Displays a comparison grid of original vs augmented images in a single figure.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get one batch
    images, targets = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate augmented versions
    augmentation_pipeline = augmentation_pipeline.to(device)
    with torch.no_grad():
        aug_images = augmentation_pipeline(images.to(device)).cpu()
    
    # Denormalization constants
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    # Setup plot: 2 rows (Original, Augmented), num_images columns
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 5))
    plt.suptitle("Image Comparison: Original (Top) vs Augmented (Bottom)", fontsize=16)

    for i in range(num_images):
        # 1. Plot Original
        orig_img = images[i] * std + mean
        orig_img = orig_img.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0, i].imshow(orig_img)
        axes[0, i].axis("off")
        
        # Get label
        label = targets[i].item()
        class_name = dataset_obj.idx_to_human_name.get(label, label) if hasattr(dataset_obj, 'idx_to_human_name') else label
        axes[0, i].set_title(class_name, fontsize=8)

        # 2. Plot Augmented
        aug_img = aug_images[i] * std + mean
        aug_img = aug_img.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[1, i].imshow(aug_img)
        axes[1, i].axis("off")

    plt.tight_layout()
    
    # Save comparison figure
    filename = "augmentation_comparison.png"
    full_save_path = os.path.join(save_path, filename)
    plt.savefig(full_save_path)
    print(f"Comparison figure saved at: {full_save_path}")
    
    plt.show()

def view_imagenet100_samples():
    """
    Main entry point to visualize the comparison.
    """
    train_loader, _, _, full_train_dataset, _ = get_imagenet_dataloaders(
        ImageNet100Dataset, DATA_ROOT_DIR
    )

    aug_pipe = get_augmentation_pipeline()
    
    # One call to see both
    show_dataset_comparison(full_train_dataset, train_loader, aug_pipe, num_images=10, save_path=DATA_ROOT_DIR)

if __name__ == "__main__": 
    view_imagenet100_samples()