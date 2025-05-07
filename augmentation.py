"""
This script defines custom image transformations that simulataneously transform image and segmentation masks.
"""
import os
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image
import matplotlib.pyplot as plt
import random

SEGMENTATION_COLOURS = {0:[0,0,0],1:[255,0,0],2:[0,253,0],3:[0,0,250], 4:[253,255,0]}
COLOUR_TO_INDEX = {tuple(v): k for k, v in SEGMENTATION_COLOURS.items()}
RGB_VALUES = np.array(list(COLOUR_TO_INDEX.keys()))  

    
class DoubleHorizontalFlip:
    """
    Apply horizontal flips to image mask pair.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        if p > self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask

class DoubleVerticalFlip:
    """
    Apply vertical flips to image and mask pair.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        if p > self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask
        
class DoubleCompose(transforms.Compose):
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
        
class MaskTransform:
    def __init__(self, tolerance=5):
        self.tolerance = tolerance

    def _match_with_tolerance(self, mask, rgb):
        rgb = np.array(rgb)
        return np.linalg.norm(mask - rgb, axis=-1) < self.tolerance
        
    def __call__(self, img):
        img = img.resize((256, 256), Image.NEAREST)
        mask = np.array(img)
        h, w, _ = mask.shape
        target = np.full((h, w), fill_value=-1, dtype=np.int64)

        # Tolerant colour match
        for rgb, idx in COLOUR_TO_INDEX.items():
            matches = self._match_with_tolerance(mask, rgb)
            # np.all(mask == rgb, axis=-1)
            target[matches] = idx
    
    
        # Handle unmatched pixels by assigning to the same class as closest matched pixel
        bad_mask = target == -1
        if bad_mask.any():
            print(f"{bad_mask.sum()} bad pixels found — applying nearest valid pixel fill.")

            # Create distance transform to nearest valid pixel
            distance, indices = distance_transform_edt(bad_mask, return_indices=True)
            # Fill each bad pixel with the class of the nearest good pixel
            target[bad_mask] = target[tuple(indices[:, bad_mask])]

        return torch.from_numpy(target).long()


class ReverseImageTransform:
    """
    Reverses transformation of image from .png to numerical tensor. Used to check dataloader.
    """
    def __call__(self, img):

        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std*img + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

        return img

class ReverseMaskTransform:
    def __init__(self):
        self.colours = np.array([SEGMENTATION_COLOURS[i] for i in range(len(SEGMENTATION_COLOURS))], dtype=np.uint8)

    def __call__(self, mask):
        """
        Args:
            mask (Tensor or ndarray): 2D array of shape (H, W) with class indices.
        
        Returns:
            PIL.Image: RGB image of shape (H, W, 3)
        """
        if hasattr(mask, 'cpu'):  # Torch Tensor
            mask = mask.cpu().numpy()
        rgb = self.colours[mask]  # shape: (H, W, 3)
        return Image.fromarray(rgb)

        

if __name__ == "__main__":
    """
    Testing the data augmentation process. Only ran when file run as a script instead of imported as a module. 
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    image_mask_transform = DoubleCompose([
        # DoubleElasticTransform(),
        DoubleHorizontalFlip(),
        DoubleVerticalFlip()
    ])
    
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = MaskTransform()

    input_dir = 'dataset/train/'
    image_dir = os.path.join(input_dir, "Images/")
    mask_dir = os.path.join(input_dir, "Labels/")
    images = os.listdir(image_dir)
    masks = os.listdir(mask_dir)

    idx = random.randint(0, len(image_dir))
    image_path = os.path.join(image_dir, images[idx])
    mask_path = os.path.join(mask_dir, masks[idx])
    image = Image.open(image_path).convert('RGB') 
    mask = Image.open(mask_path) 

    image_t, mask_t = image_mask_transform(image, mask)
    image_t = image_transform(image)
    mask_t = mask_transform(mask)

    reverse_transform = ReverseImageTransform()
    reverse_mask_transform = ReverseMaskTransform()

    image_t = reverse_transform(image_t)
    mask_t = reverse_mask_transform(mask_t)

    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(image)
    ax.set_title('Image')
    ax = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(image_t)
    ax.set_title('Transformed Image')
    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(mask)
    ax.set_title('Label')
    ax = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(mask_t)
    ax.set_title('Transformed Label')
    fig.tight_layout()

    plt.show()
