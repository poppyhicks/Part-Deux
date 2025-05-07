import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from augmentation import MaskTransform


class SegmentationDataset(Dataset):
    """
    Loads and pairs image-mask data.
    """
    def __init__(self, input_dir, image_mask_transform=None, image_transform = None, mask_transform=None):

        self.image_mask_transform = image_mask_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_dir = os.path.join(input_dir, "Images/")
        self.mask_dir = os.path.join(input_dir, "Labels/")
        self.images = os.listdir(self.image_dir)
        self.masks = os.listdir(self.mask_dir)

    def __len__(self):
        """
        Returns the number of samples in our dataset
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset.  
        """
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(image_path)
        mask = Image.open(mask_path) 

        if self.image_mask_transform:
            image, mask = self.image_mask_transform(image, mask)    
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask
    

if __name__ == "__main__":
    """
    Testing: finding proportion of classes in labels to generate class weights for loss.
    """
    mask_transform = MaskTransform()
    train_dataset = SegmentationDataset('dataset/train/', image_mask_transform=None, image_transform=None, mask_transform=mask_transform)
    counts, colours = [], []
    for _, mask in train_dataset:
        mask_array = mask.numpy().flatten()
        colour, count = np.unique(mask_array, return_counts=True, axis = 0)
        counts.append(count)
        colours.append(colour)
    counts = np.concatenate(counts).ravel()
    colours = np.concatenate(colours).ravel()

    class_counts = []
    for i in range(5):
        indices = np.where(colours == i)[0]
        class_count = counts[indices]
        class_counts.append(sum(class_count))
    proportion = np.array(class_counts)/sum(counts)
    print('Pixels of each class:' , class_counts)
    print('Proportion: ', proportion)