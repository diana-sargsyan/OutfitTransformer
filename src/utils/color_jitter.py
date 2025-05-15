import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class OutfitColorJitterAugmentation:
    """
    Color jitter augmentation for outfit compatibility prediction.
    
    Applies random perturbations to brightness, contrast, saturation, and hue
    for each item in an outfit independently to reduce color uniformity bias.
    """
    def __init__(self, 
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 saturation_range=(0.7, 1.3),
                 hue_range=(-0.1, 0.1)):
        """
        Initialize color jitter augmentation with specified parameter ranges.
        
        Args:
            brightness_range (tuple): Range for brightness adjustment factor
            contrast_range (tuple): Range for contrast adjustment factor
            saturation_range (tuple): Range for saturation adjustment factor
            hue_range (tuple): Range for hue shift in normalized space
        """
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness_range,
            contrast=contrast_range,
            saturation=saturation_range,
            hue=hue_range
        )
        
        # Default transformation pipeline to be applied before and after jitter
        self.base_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, outfit_images):
        """
        Apply color jitter augmentation to each item in an outfit independently.
        
        Args:
            outfit_images (list): List of PIL images representing outfit items
            
        Returns:
            list: Augmented outfit images as tensors
        """
        augmented_outfit = []
        
        for image in outfit_images:
            # Convert tensor to PIL image if needed
            if isinstance(image, torch.Tensor):
                if image.shape[0] == 3:  # If image is already a tensor with shape [3, H, W]
                    image = transforms.ToPILImage()(image)
                else:
                    # Handle grayscale or other formats if needed
                    continue
            
            # Apply color jitter independently to each item
            augmented_image = self.color_jitter(image)
            
            # Convert back to tensor and normalize
            augmented_image = self.base_transforms(augmented_image)
            
            augmented_outfit.append(augmented_image)
            
        return augmented_outfit


class PolyvoreOutfitDataset(torch.utils.data.Dataset):
    """
    Dataset class for Polyvore outfits with color jitter augmentation.
    This is a simplified example that should be adapted to your actual dataset structure.
    """
    def __init__(self, outfit_data, image_dir, transform=None, apply_jitter=True):
        """
        Initialize dataset with optional color jitter augmentation.
        
        Args:
            outfit_data (list): List of outfit metadata
            image_dir (str): Directory containing outfit images
            transform (callable, optional): Transform to apply to images
            apply_jitter (bool): Whether to apply color jitter augmentation
        """
        self.outfit_data = outfit_data
        self.image_dir = image_dir
        self.transform = transform
        self.apply_jitter = apply_jitter
        
        if self.apply_jitter:
            self.jitter_transform = OutfitColorJitterAugmentation()
        
    def __len__(self):
        return len(self.outfit_data)
    
    def __getitem__(self, idx):
        outfit = self.outfit_data[idx]
        
        # Load images for each item in the outfit
        outfit_images = []
        for item_id in outfit['items']:
            image_path = f"{self.image_dir}/{item_id}.jpg"
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            outfit_images.append(image)
        
        # Apply jitter augmentation if enabled (only during training)
        if self.apply_jitter and self.mode == 'train':
            outfit_images = self.jitter_transform(outfit_images)
        
        # Create compatible features and labels as needed by OutfitTransformer
        compatibility_label = outfit.get('compatibility', 1.0)
        
        return {
            'outfit_images': outfit_images,
            'item_ids': outfit['items'],
            'compatibility': torch.tensor(compatibility_label, dtype=torch.float)
        }


# Example usage in the main training script
def create_data_loaders(config):
    """
    Create data loaders for training and validation with color jitter augmentation.
    
    Args:
        config: Configuration for dataset and dataloaders
        
    Returns:
        tuple: Training and validation dataloaders
    """
    # Base transforms without jitter (for validation and test)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create train dataset with jitter augmentation
    train_dataset = PolyvoreOutfitDataset(
        outfit_data=config.train_data,
        image_dir=config.image_dir,
        transform=base_transform,
        apply_jitter=True
    )
    train_dataset.mode = 'train'  # Set mode to enable jitter
    
    # Create validation dataset without jitter
    val_dataset = PolyvoreOutfitDataset(
        outfit_data=config.val_data,
        image_dir=config.image_dir,
        transform=base_transform,
        apply_jitter=False
    )
    val_dataset.mode = 'val'  # Set mode to disable jitter
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    return train_loader, val_loader